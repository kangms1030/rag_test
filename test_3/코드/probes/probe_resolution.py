"""P0-A: 해상도/타일링이 VLM 표·도표 오독을 고치는지 조건별 전사 정확도로 측정.

가설(계획 F3/F4): 오독의 원인이 (a) 저해상도 크롭인지, (b) Ollama 런타임의 비전 입력 뭉갬
(#10392 pan&scan 미구현, #15626 max_soft_tokens=280)인지 분리한다. (b)가 원인이면 해상도를
올려도 안 고쳐지고, 타일링(이미지당 개별 인코딩=수동 pan&scan)만 효과가 있을 것이다.

대상: vlm_probe에서 정답 페이지를 정확히 봤는데 오독한 사례 5 + 정답(대조) 1.
조건: 1) baseline(test_2가 실제 보내는 이미지, 2048 축소)
      2) page_hidpi(PDF 페이지 전체 350DPI, 축소 없이 전송)
      3) page_tiled(300DPI 페이지를 2x2 타일 4장으로 전송)
      4) crop_hidpi(표 bbox를 400DPI로 재렌더, 표 없으면 skip)
채점: gold 토큰 포함 & 기지 오독(bad) 토큰 미포함 => 정답.

결과: test_3/probes/results/p0a_resolution.json
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import fitz  # PyMuPDF
import ollama
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "rag_test" / "test_2" / "rag2" / "cache" / "parsed"
DOCS_DIR = ROOT / "rag_test" / "test_2" / "데이터 카탈로그 작업 파일"
OUT = Path(__file__).resolve().parent / "results" / "p0a_resolution.json"
MODEL = "gemma4:12b"
SEND_MAX_SIDE = 2048  # baseline 재현용(test_2 answer_image_max_side)

# 대상 페이지 + 기지 gold/misread 토큰 (fact 문서 §4.3B / vlm_probe REPORT §2)
CASES = [
    {"qid": "vp_006", "slug": "mdm-argos-edu-v1-5-6061726f", "page": 83,
     "gold": ["R54KB00TXWF"], "bad": ["R54K000TWW", "R54K000"], "note": "시리얼 오독"},
    {"qid": "vp_009", "slug": "0-1v-20p-2023-a874b961", "page": 13,
     "gold": ["개요", "갤럭시"], "bad": ["캐너"], "note": "WiFi 개요 360->캐너 / 갤럭시 앱 스토어 누락"},
    {"qid": "vp_010", "slug": "0-1v-20p-2023-a874b961", "page": 11,
     "gold": ["집선"], "bad": ["확장"], "note": "무선 집선 스위치->확장"},
    {"qid": "vp_014", "slug": "weiss-de-001-3-v0-82-f69ab0fd", "page": 71,
     "gold": ["인증"], "bad": [], "note": "인증->운영"},
    {"qid": "vp_007", "slug": "mdm-argos-edu-v1-5-6061726f", "page": 88,
     "gold": ["135"], "bad": [], "note": "135건->5건 (숫자, 참고용)"},
    {"qid": "vp_011", "slug": "23-b3bffb22", "page": 22,
     "gold": ["등록", "승인"], "bad": [], "note": "대조군(원래 정답)"},
]

PROMPT = """이 이미지는 문서의 한 페이지(또는 표)다. 이미지에 보이는 표/텍스트의 내용을
행과 열 구조를 유지하며 **글자 그대로** 옮겨 적어라(전사). 없는 내용은 지어내지 말고,
숫자·영문 시리얼·고유명사는 특히 정확히 옮겨라. 표만 있으면 표만, 도표면 라벨을 옮겨라."""


def _find_pdf(slug: str) -> Path | None:
    man = CACHE / slug / "manifest.json"
    m = json.loads(man.read_text(encoding="utf-8"))
    # abs_path가 stale일 수 있으니 document_name으로 재탐색
    name = m["document_name"]
    for p in DOCS_DIR.rglob("*.pdf"):
        if p.name == name:
            return p
    ap = Path(m.get("abs_path", ""))
    return ap if ap.exists() else None


def _content_list(slug: str) -> list[dict]:
    d = CACHE / slug / "mineru" / slug / "auto" / f"{slug}_content_list.json"
    if not d.exists():
        return []
    return json.loads(d.read_text(encoding="utf-8"))


def _table_bbox(slug: str, page: int) -> list[int] | None:
    """content_list에서 해당 페이지의 대표 표 bbox(0~1000 정규화) 반환."""
    items = [it for it in _content_list(slug) if it.get("page_idx") == page - 1]
    tables = [it for it in items if it.get("type") == "table" and it.get("bbox")]
    if not tables:
        return None
    return max(tables, key=lambda it: (it["bbox"][2] - it["bbox"][0]) * (it["bbox"][3] - it["bbox"][1]))["bbox"]


def _pix_to_png_bytes(pix: fitz.Pixmap) -> bytes:
    return pix.tobytes("png")


def _resize_bytes(img_bytes: bytes, max_side: int) -> bytes:
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(1.0, max_side / max(w, h))
        if s < 1.0:
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


def _tiles_2x2(img_bytes: bytes) -> list[bytes]:
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        w, h = im.size
        mx, my = w // 2, h // 2
        boxes = [(0, 0, mx, my), (mx, 0, w, my), (0, my, mx, h), (mx, my, w, h)]
        out = []
        for b in boxes:
            buf = io.BytesIO()
            im.crop(b).save(buf, format="JPEG", quality=92)
            out.append(buf.getvalue())
        return out


def render_conditions(case: dict) -> dict[str, list[bytes]]:
    """조건별 전송 이미지 바이트 리스트."""
    slug, page = case["slug"], case["page"]
    man = json.loads((CACHE / slug / "manifest.json").read_text(encoding="utf-8"))
    pg = next(p for p in man["pages"] if p["page_number"] == page)
    conds: dict[str, list[bytes]] = {}

    # 1) baseline: test_2가 실제 보내는 이미지(table_crop 우선, 없으면 page png) -> 2048 축소
    base_img = pg.get("table_crop_path") or pg.get("page_image_path")
    bp = Path(base_img)
    if bp.exists():
        conds["baseline"] = [_resize_bytes(bp.read_bytes(), SEND_MAX_SIDE)]

    pdf = _find_pdf(slug)
    if pdf is not None:
        doc = fitz.open(str(pdf))
        try:
            fp = doc[page - 1]
            # 2) page_hidpi 350DPI 전체(축소 없이, 단 Ollama 안전상 2048 캡은 두되 고DPI 원본을 캡)
            pix = fp.get_pixmap(matrix=fitz.Matrix(350 / 72, 350 / 72))
            conds["page_hidpi350"] = [_resize_bytes(_pix_to_png_bytes(pix), SEND_MAX_SIDE)]
            # 3) page_tiled: 300DPI 페이지를 2x2 타일 4장
            pix2 = fp.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
            conds["page_tiled2x2"] = _tiles_2x2(_pix_to_png_bytes(pix2))
            # 4) crop_hidpi: 표 bbox 400DPI 재렌더
            bbox = _table_bbox(slug, page)
            if bbox:
                r = fp.rect
                clip = fitz.Rect(bbox[0] / 1000 * r.width, bbox[1] / 1000 * r.height,
                                 bbox[2] / 1000 * r.width, bbox[3] / 1000 * r.height)
                pix3 = fp.get_pixmap(matrix=fitz.Matrix(400 / 72, 400 / 72), clip=clip)
                conds["crop_hidpi400"] = [_resize_bytes(_pix_to_png_bytes(pix3), SEND_MAX_SIDE)]
        finally:
            doc.close()
    return conds


def ask_vlm(images: list[bytes]) -> str:
    b64 = [base64.b64encode(im).decode() for im in images]
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": b64}],
        options={"num_ctx": 8192, "temperature": 0.0},
        keep_alive="10m",
    )
    return resp["message"]["content"]


def score(text: str, gold: list[str], bad: list[str]) -> dict:
    g = [tok for tok in gold if tok in text]
    b = [tok for tok in bad if tok in text]
    return {"gold_found": g, "bad_found": b,
            "correct": (len(g) == len(gold) and not b)}


def main() -> None:
    out = {"model": MODEL, "cases": []}
    for case in CASES:
        print(f"\n[{case['qid']}] {case['note']} (slug={case['slug']} p{case['page']})")
        try:
            conds = render_conditions(case)
        except Exception as e:
            print(f"    렌더 실패: {e}")
            out["cases"].append({**case, "error": str(e)})
            continue
        rec = {**case, "conditions": {}}
        for cname, imgs in conds.items():
            t0 = time.time()
            try:
                raw = ask_vlm(imgs)
            except Exception as e:
                rec["conditions"][cname] = {"error": str(e)}
                print(f"    {cname:16s} 호출 실패: {e}")
                continue
            sc = score(raw, case["gold"], case["bad"])
            rec["conditions"][cname] = {
                "n_images": len(imgs), "wall_s": round(time.time() - t0, 1),
                **sc, "raw_head": raw[:220],
            }
            mark = "O" if sc["correct"] else "X"
            print(f"    {cname:16s} [{mark}] gold={sc['gold_found']} bad={sc['bad_found']} ({len(imgs)}img)")
        out["cases"].append(rec)

    # 조건별 정답 수 집계
    cond_names = ["baseline", "page_hidpi350", "page_tiled2x2", "crop_hidpi400"]
    tally = {c: {"correct": 0, "total": 0} for c in cond_names}
    for rec in out["cases"]:
        for c in cond_names:
            r = rec.get("conditions", {}).get(c)
            if r and "correct" in r:
                tally[c]["total"] += 1
                tally[c]["correct"] += int(r["correct"])
    out["tally"] = tally
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P0-A 조건별 정답(전사 정확) 집계 ===")
    for c in cond_names:
        t = tally[c]
        print(f"  {c:16s} {t['correct']}/{t['total']}")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()

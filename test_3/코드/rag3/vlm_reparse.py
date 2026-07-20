"""Phase 3: figure 페이지 MinerU vlm-engine 재파싱 결과를 pipeline 캐시에 병합.

P0-B(PHASE3_P0B_REPORT.md) 결론: 벡터 도표/구성도 페이지는 현행 pipeline OCR이 텍스트를
거의 못 뽑지만(gold 0), MinerU vlm-engine은 라벨을 구조화 텍스트(mermaid 포함)로 추출한다.
반대로 래스터 UI 표는 vlm-engine이 회귀시키므로, **figure 페이지만** 재파싱해 병합한다
(page-type routing). 래스터 표/일반 텍스트 페이지는 건드리지 않아 회귀가 없다.

병합 방식(비파괴적, parsed_v25 별도 캐시):
  1) content_list에 해당 page_idx의 `type:"text"` 블록을 추가 → chunking이 텍스트 청크 생성.
     (chunking은 image 블록을 스킵하므로 vlm의 mermaid content를 text 블록으로 승격해야 검색된다.)
  2) manifest의 해당 페이지 text/char_count 갱신 + page_type="figure"→"text"
     → retrieve._route_v2가 vision이 아니라 text 경로를 택하게 함(도표를 text로 답변, 오독 제거).
"""
from __future__ import annotations

import glob
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_NOISE = {"footer", "header", "page_number"}
_VLM_FLAG = "_vlm_reparse"
_ORIG = "_vlm_orig_text"


def _flatten_mermaid(block: str) -> str:
    """mermaid 그래프 소스에서 노드/서브그래프 라벨("..." 리터럴)만 뽑아 평문으로.

    12b는 원시 mermaid 코드 블록을 받으면 빈 응답을 내는 경향(P0-C)이 있어, 도표의
    텍스트 내용만 남긴다. edge/문법은 버리고 라벨의 의미 텍스트만 보존한다.
    """
    labels = re.findall(r'\["([^"]*)"\]|\("([^"]*)"\)|\{"([^"]*)"\}|>"([^"]*)"\]', block)
    out = []
    for tup in labels:
        s = next((x for x in tup if x), "").replace("\\n", " ").strip()
        if s and s not in out:
            out.append(s)
    return " · ".join(out)


def _demermaid(text: str) -> str:
    """```mermaid ...``` 펜스를 평탄화된 라벨 텍스트로 치환. 나머지는 그대로."""
    return re.sub(r"```mermaid(.*?)```", lambda m: _flatten_mermaid(m.group(1)),
                  text, flags=re.DOTALL)


def find_figure_pages(content_list: list[dict], *, txt_max: int = 200) -> list[int]:
    """이미지 블록이 있고 페이지 텍스트가 txt_max 미만인 page_idx(0-based) 목록."""
    pages: dict[int, dict] = {}
    for it in content_list:
        pi = it.get("page_idx")
        if pi is None:
            continue
        d = pages.setdefault(int(pi), {"img": 0, "txt": 0})
        if it.get("type") in ("image", "figure"):
            d["img"] += 1
        d["txt"] += len(it.get("text", "") or "") + len(it.get("table_body", "") or "")
    return sorted(pi for pi, d in pages.items() if d["img"] > 0 and d["txt"] < txt_max)


def extract_vlm_page_text(blocks: list[dict]) -> str:
    """한 페이지의 vlm content_list 블록들에서 검색·답변에 쓸 텍스트를 조립.

    text 블록 + 표(caption+body) + image(caption + content: mermaid/설명)를 합친다.
    footer/header/page_number는 제외.
    """
    parts: list[str] = []
    for it in blocks:
        t = it.get("type")
        if t in _NOISE:
            continue
        if t == "text":
            parts.append(it.get("text", "") or "")
        elif t == "table":
            parts.append(" ".join(it.get("table_caption", []) or []))
            parts.append(it.get("table_body", "") or "")
            parts.append(" ".join(it.get("table_footnote", []) or []))
        elif t in ("image", "figure"):
            parts.append(" ".join(it.get("image_caption", []) or []))
            c = it.get("content")
            if isinstance(c, str):
                parts.append(_demermaid(c))
    return "\n".join(p for p in parts if p and p.strip()).strip()


def vlm_texts_by_page(vlm_content_list: list[dict]) -> dict[int, str]:
    """vlm 산출 content_list(결합 PDF 1개) -> {mini_page_idx(0-based): 조립 텍스트}."""
    by_page: dict[int, list[dict]] = {}
    for it in vlm_content_list:
        by_page.setdefault(int(it.get("page_idx", 0)), []).append(it)
    return {pi: extract_vlm_page_text(blocks) for pi, blocks in by_page.items()}


def _content_list_path(parsed_dir: Path, slug: str) -> Path | None:
    hits = glob.glob(str(parsed_dir / slug / "mineru" / "*" / "auto" / "*_content_list.json"))
    return Path(hits[0]) if hits else None


def merge_doc(parsed_dir: Path, slug: str, page_texts: dict[int, str]) -> dict:
    """parsed_dir/<slug>의 content_list와 manifest를 in-place 증강. page_texts: {1-based page: text}.

    반복 실행 안전(_vlm_reparse 블록을 먼저 제거 후 재삽입, page_type 이미 text면 유지).
    """
    stats = {"slug": slug, "pages": 0, "chars": 0, "skipped_empty": 0}
    clp = _content_list_path(parsed_dir, slug)
    if clp is None:
        logger.warning("[%s] content_list 없음 -> 병합 생략", slug)
        return stats
    cl = json.loads(clp.read_text(encoding="utf-8"))
    cl = [b for b in cl if not b.get(_VLM_FLAG)]  # 이전 병합 흔적 제거(멱등)
    for page, vtext in sorted(page_texts.items()):
        if not vtext or not vtext.strip():
            stats["skipped_empty"] += 1
            continue
        cl.append({"type": "text", "text": vtext, "text_level": 2,
                   "page_idx": page - 1, _VLM_FLAG: True})
        stats["pages"] += 1
        stats["chars"] += len(vtext)
    clp.write_text(json.dumps(cl, ensure_ascii=False), encoding="utf-8")

    manp = parsed_dir / slug / "manifest.json"
    man = json.loads(manp.read_text(encoding="utf-8"))
    for p in man.get("pages", []):
        pn = p.get("page_number")
        if pn in page_texts and page_texts[pn].strip():
            # 멱등: 원본 텍스트를 1회 보존해두고 매번 그 위에 재구성(재병합 시 이중추가 방지)
            if _ORIG not in p:
                p[_ORIG] = p.get("text", "") or ""
            base = (p[_ORIG] or "").strip()
            add = page_texts[pn].strip()
            # 도표 페이지는 깨끗한 vlm 텍스트를 앞세워 12b 빈응답/오독을 줄임
            p["text"] = (add + "\n" + base).strip() if base else add
            p["char_count"] = len(p["text"])
            p["page_type"] = "text"  # 이제 실제 텍스트 보유 -> text 라우팅
    manp.write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    return stats

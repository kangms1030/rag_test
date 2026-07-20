"""P0-C: catalog 실패 6건의 빈 응답/거절 원인 진단 (num_ctx 오버플로 vs 1순위 방해물).

가설(계획 F1/F8): 정답 페이지가 top3 컨텍스트 안에 있었는데도 빈 응답/"확인 불가"가 난 건
PROMPT_TEXT_ANSWER가 [지시문 -> 질문 -> 근거 -> "답변:"] 순서라서, 근거(표 HTML 포함)가 커지면
num_ctx 8192를 넘겨 Ollama가 프롬프트 앞부분(지시문/질문)을 잘라버렸기 때문일 수 있다.

방법(원본 파이프라인 답변 로직을 그대로 재현):
- 각 실패 qid의 evidence(문서+페이지 순서)를 캐시 manifest에서 텍스트로 복원 -> 원본 answer.PROMPT_TEXT_ANSWER 조립
- 조건 A: num_ctx 8192 (원본) -> prompt_eval_count, 답변 공백 여부
- 조건 B: num_ctx 16384 -> 같은 프롬프트로 답변이 살아나는지
- 조건 C: 페이지 순서를 정답페이지 우선으로 재배열 후 num_ctx 8192
tokenizer로 대략 토큰수도 병기. 답변 품질(정오)은 여기서 판정하지 않고 "공백/거절 여부"만 본다.

결과: test_3/probes/results/p0c_ctx_overflow.json
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import ollama

ROOT = Path(__file__).resolve().parents[2]  # 챗봇/
sys.path.insert(0, str(ROOT / "test_3"))
from rag3.answer import PROMPT_TEXT_ANSWER, _format_context, _NO_DOC_ANSWER  # noqa: E402

CACHE = ROOT / "rag_test" / "test_2" / "rag2" / "cache" / "parsed"
PROBE_RESULTS = ROOT / "test12_total_test" / "test2_vlm_probe" / "results" / "catalog"
DATASET = ROOT / "test12_total_test" / "test2_vlm_probe" / "vlm_probe_dataset.json"
OUT = Path(__file__).resolve().parent / "results" / "p0c_ctx_overflow.json"

# catalog 모드 실패 6건 (계획 §신규진단): 빈 응답 3 + 확인불가 2 + 게이트거절 1
FAIL_QIDS = ["vp_002", "vp_006", "vp_007", "vp_010", "vp_014", "vp_005"]
MODEL = "gemma4:12b"


def build_docmap() -> dict[str, dict[int, dict]]:
    """document_name -> {page_number -> page dict(text 포함)}."""
    docmap: dict[str, dict[int, dict]] = {}
    for man in CACHE.glob("*/manifest.json"):
        m = json.loads(man.read_text(encoding="utf-8"))
        docmap[m["document_name"]] = {p["page_number"]: p for p in m["pages"]}
    return docmap


def load_expected() -> dict[str, dict]:
    ds = json.loads(DATASET.read_text(encoding="utf-8"))
    items = ds if isinstance(ds, list) else ds.get("questions", ds.get("items", []))
    return {q["id"]: q for q in items}


def rough_tokens(text: str) -> int:
    """대략 토큰수(한국어 혼합): 공백분할 + CJK 글자수의 절반 가산 근사."""
    words = len(text.split())
    cjk = sum(1 for c in text if "가" <= c <= "힣" or "一" <= c <= "鿿")
    return words + cjk // 2


def call(prompt: str, num_ctx: int) -> dict:
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": num_ctx, "temperature": 0.0},
        keep_alive="10m",
    )
    content = resp["message"]["content"].strip()
    return {
        "num_ctx": num_ctx,
        "prompt_eval_count": resp.get("prompt_eval_count"),
        "eval_count": resp.get("eval_count"),
        "answer_empty": (content == ""),
        "answer_is_nodoc": (_NO_DOC_ANSWER in content or "확인할 수 없" in content or "확인 불가" in content),
        "answer_head": content[:180],
    }


def main() -> None:
    docmap = build_docmap()
    expected = load_expected()
    out: dict = {"model": MODEL, "cases": []}

    for qid in FAIL_QIDS:
        rp = PROBE_RESULTS / f"{qid}_result.json"
        if not rp.exists():
            print(f"[{qid}] 결과 파일 없음, 건너뜀")
            continue
        res = json.loads(rp.read_text(encoding="utf-8"))
        question = res["question"]
        evidence = res.get("evidence", [])  # [{document_name, page_number, ...}] 순서 = 컨텍스트 순서
        exp = expected.get(qid, {})
        exp_pages = exp.get("expected_pages") or []
        answer_page = exp_pages[0] if exp_pages else None

        # 원본 파이프라인이 본 페이지 순서 그대로 텍스트 복원
        pages = []
        for ev in evidence:
            dn, pn = ev["document_name"], ev["page_number"]
            pg = docmap.get(dn, {}).get(pn)
            if pg is None:
                continue
            pages.append({"document_name": dn, "page_number": pn, "text": pg["text"], "chars": len(pg["text"])})

        case: dict = {
            "qid": qid,
            "question": question,
            "orig_answer_path": res.get("answer_path"),
            "orig_final_answer": res.get("final_answer"),
            "answer_page_expected": answer_page,
            "pages_seen": [(p["document_name"][:22], p["page_number"], p["chars"]) for p in pages],
        }
        if not pages:
            case["note"] = "복원 페이지 없음(게이트 none일 가능성)"
            out["cases"].append(case)
            print(f"[{qid}] 페이지 복원 실패 (orig_path={res.get('answer_path')})")
            continue

        context = _format_context(pages)
        prompt = PROMPT_TEXT_ANSWER.format(question=question, context=context)
        case["context_chars"] = len(context)
        case["prompt_chars"] = len(prompt)
        case["prompt_rough_tokens"] = rough_tokens(prompt)

        # 조건 A: num_ctx 8192 (원본 재현)
        print(f"[{qid}] 프롬프트 {len(prompt)}자(~{case['prompt_rough_tokens']}tok) 재현 중...")
        case["A_ctx8192"] = call(prompt, 8192)
        # 조건 B: num_ctx 16384
        case["B_ctx16384"] = call(prompt, 16384)
        # 조건 C: 정답페이지 우선 재배열 + 8192
        if answer_page is not None and any(p["page_number"] == answer_page for p in pages):
            reordered = sorted(pages, key=lambda p: 0 if p["page_number"] == answer_page else 1)
            ctx2 = _format_context(reordered)
            prompt2 = PROMPT_TEXT_ANSWER.format(question=question, context=ctx2)
            case["C_reorder_ctx8192"] = call(prompt2, 8192)
        else:
            case["C_reorder_ctx8192"] = {"skipped": "정답페이지 미상 또는 컨텍스트에 없음"}

        a, b = case["A_ctx8192"], case["B_ctx16384"]
        print(f"    A(8192): peval={a['prompt_eval_count']} empty={a['answer_empty']} nodoc={a['answer_is_nodoc']}")
        print(f"    B(16384): peval={b['prompt_eval_count']} empty={b['answer_empty']} nodoc={b['answer_is_nodoc']}")
        out["cases"].append(case)

    # 종합 판정
    overflow_hits = []
    fixed_by_16k = []
    fixed_by_reorder = []
    for c in out["cases"]:
        a = c.get("A_ctx8192")
        b = c.get("B_ctx16384")
        cc = c.get("C_reorder_ctx8192")
        if not a:
            continue
        # 오버플로 신호: prompt_eval_count가 8192에 근접(>=8000)하거나 A 실패인데 B 성공
        if a.get("prompt_eval_count") and a["prompt_eval_count"] >= 8000:
            overflow_hits.append(c["qid"])
        a_fail = a["answer_empty"] or a["answer_is_nodoc"]
        b_ok = b and not (b["answer_empty"] or b["answer_is_nodoc"])
        if a_fail and b_ok:
            fixed_by_16k.append(c["qid"])
        if a_fail and isinstance(cc, dict) and "skipped" not in cc and not (cc["answer_empty"] or cc["answer_is_nodoc"]):
            fixed_by_reorder.append(c["qid"])
    out["verdict"] = {
        "prompt_eval>=8000 (오버플로 근접)": overflow_hits,
        "16k로 살아난 케이스": fixed_by_16k,
        "정답페이지 우선재배열로 살아난 케이스": fixed_by_reorder,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P0-C 판정 ===")
    for k, v in out["verdict"].items():
        print(f"  {k}: {v}")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()

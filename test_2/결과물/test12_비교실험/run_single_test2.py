"""test_2(rag2) 단일 (mode, question) 실행기.

rag2 소스는 수정하지 않고 import만 한다(test_2_timecost/README.md의 계약과 동일). ingest
때 969페이지 전량을 MinerU로 사전 파싱·색인해 두었으므로 ask 경로는 인덱스/캐시에 아무것도
쓰지 않는다 — 실행 순서와 무관하며 재-ingest도 필요 없다(test_2_timecost/results/REPORT.md
"baseline 정합성 대조"에서 이미 실측 확인됨). 그래도 실행 간 완전한 격리를 위해 이 스크립트도
독립 서브프로세스로 매 (mode, question)마다 새로 띄운다.

산출물은 --result-out 경로에만 쓴다. test_2/ 아래에는 어떤 파일도 새로 생기지 않는다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TEST2_DIR = _HERE.parent / "test_2"
_TEST2_TIMECOST_DIR = _HERE.parent / "test_2_timecost"
sys.path.insert(0, str(_TEST2_DIR))
sys.path.insert(0, str(_TEST2_TIMECOST_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_single_test2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["catalog", "no_catalog"])
    ap.add_argument("--qid", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--expected-documents", default="[]")
    ap.add_argument("--expected-pages", default="[]")
    ap.add_argument("--expected-keywords", default="[]")
    ap.add_argument("--result-out", required=True)
    args = ap.parse_args()

    from rag2.answer import generate_answer
    from rag2.config import load_config
    from rag2.metrics import RunMetrics, run_metrics
    from rag2.models import get_backend
    from rag2.retrieve import run_retrieval

    from retrieve_no_catalog import retrieve_no_catalog

    retrieval_fn = run_retrieval if args.mode == "catalog" else retrieve_no_catalog

    config = load_config()  # rag2 내장 config.yaml 그대로 사용 (경로는 test_2/rag2 기준)
    backend = get_backend(config)

    question = args.question.strip()
    run_metric = RunMetrics()

    error: str | None = None
    retrieval = None
    answer_result: dict = {"final_answer": "", "answer_path": "none", "evidence": []}

    t_start = time.monotonic()
    t0 = t1 = t2 = t_start
    try:
        with run_metrics(run_metric):
            t0 = time.monotonic()
            retrieval = retrieval_fn(question, config, backend)
            t1 = time.monotonic()
            from rag2.metrics import record_timing

            record_timing("retrieve", t1 - t0)
            answer_result = generate_answer(question, retrieval, backend, config)
            t2 = time.monotonic()
            record_timing("answer", t2 - t1)
            record_timing("total", t2 - t0)
    except Exception as e:
        logger.exception("실행 실패: %s", e)
        error = f"{type(e).__name__}: {e}"
        t2 = time.monotonic()
        if t1 == t_start:
            t1 = t2

    total = time.monotonic() - t_start
    metrics_dict = run_metric.to_dict()

    expected_documents = json.loads(args.expected_documents)
    expected_pages = json.loads(args.expected_pages)
    expected_keywords = json.loads(args.expected_keywords)

    selected_doc_names = [d["document_name"] for d in retrieval.selected_documents] if retrieval else []
    selected_page_numbers = [p["page_number"] for p in retrieval.selected_pages] if retrieval else []

    answer_text = answer_result.get("final_answer", "") or ""
    answer_lower = answer_text.lower()
    kw_found = [k for k in expected_keywords if k.lower() in answer_lower]
    kw_missing = [k for k in expected_keywords if k not in kw_found]

    evidence_out = []
    for e in answer_result.get("evidence", []):
        evidence_out.append(
            {
                "document_name": e.get("document_name"),
                "page_number": e.get("page_number"),
                "page_image_path": e.get("page_image_path", ""),
                "table_crop_path": e.get("table_crop_path", ""),
            }
        )

    result = {
        "pipeline": "test2",
        "mode": args.mode,
        "qid": args.qid,
        "question": question,
        "error": error,
        "final_answer": answer_text,
        "answer_path": answer_result.get("answer_path", "none"),
        "abstained": answer_result.get("answer_path", "none") == "none",
        "skip_reason": answer_result.get("skip_reason", getattr(retrieval, "route_reason", "") if retrieval else ""),
        "selected_documents": selected_doc_names,
        "selected_pages": selected_page_numbers,
        "doc_match": (any(d in selected_doc_names for d in expected_documents) if expected_documents else None),
        "page_match": (any(p in selected_page_numbers for p in expected_pages) if expected_pages else None),
        "keyword_recall": (round(len(kw_found) / len(expected_keywords), 3) if expected_keywords else None),
        "keywords_found": kw_found,
        "keywords_missing": kw_missing,
        "is_answer_supported": None,  # rag2는 별도 verify 단계가 없음(README 설계 차이)
        "stage_timings_seconds": {
            "retrieve": round(t1 - t0, 3),
            "answer": round(t2 - t1, 3),
            "total": round(total, 3),
        },
        "model_calls": {
            "embed_calls": metrics_dict["embed_calls"],
            "text_answer_calls": metrics_dict["text_answer_calls"],
            "vision_answer_calls": metrics_dict["vision_answer_calls"],
            "llm_calls_total": metrics_dict["text_answer_calls"],
            "vlm_calls_total": metrics_dict["vision_answer_calls"],
            "total_model_calls": metrics_dict["total_model_calls"],
        },
        "evidence": evidence_out,
    }

    out_path = Path(args.result_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({"qid": args.qid, "mode": args.mode, "total_seconds": result["stage_timings_seconds"]["total"], "error": error}, ensure_ascii=False))


if __name__ == "__main__":
    main()

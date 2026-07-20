"""test_1(rag_catalog_experiment) 단일 (mode, question) 냉시작 실행기.

run_final_experiment.py가 매 실행 직전 post-ingest 스냅샷을 --index-dir/--cache-dir/
--output-dir(보통 _work_test1/*)로 복원한 뒤, 이 스크립트를 독립 서브프로세스로 띄운다.
프로세스 경계가 곧 캐시/컨텍스트 격리 경계다 — in-memory 캐시나 contextvar가 실행 간
새지 않도록 매 (mode, question)마다 새 인터프리터를 쓴다.

rag_catalog_experiment 소스는 전혀 수정하지 않는다. 단계별(query_analyzer/page_summary/
visual_chunk) 소요시간은 retrieval 모듈이 참조하는 함수 이름을 이 스크립트 안에서만
monkeypatch해 얻는다 — retrieval.py가 이 이름들을 "자기 모듈 전역"으로 호출하므로
(직접 정의했거나 `from .x import y`로 들여온 이름 바인딩) 여기서 재바인딩하면 그대로 반영된다.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TEST1_PKG_DIR = _HERE.parent / "test_1차"
sys.path.insert(0, str(_TEST1_PKG_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_single_test1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["catalog", "no_catalog"])
    ap.add_argument("--qid", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--expected-documents", default="[]")
    ap.add_argument("--expected-pages", default="[]")
    ap.add_argument("--expected-keywords", default="[]")
    ap.add_argument("--result-out", required=True)
    args = ap.parse_args()

    from rag_catalog_experiment import retrieval as retrieval_mod
    from rag_catalog_experiment.answer import build_page_evidence, generate_answer, verify_answer
    from rag_catalog_experiment.config import load_config
    from rag_catalog_experiment.metrics import RunMetrics, run_metrics
    from rag_catalog_experiment.models import get_backend
    from rag_catalog_experiment.utils import new_run_id

    config = load_config(
        _TEST1_PKG_DIR / "rag_catalog_experiment" / "config.yaml",
        overrides={
            "index_dir": args.index_dir,
            "cache_dir": args.cache_dir,
            "output_dir": args.output_dir,
        },
    )
    config.ensure_dirs()
    backend = get_backend(config)

    # --- 단계별 소요시간 계측: retrieval.py가 참조하는 이름을 직접 재바인딩 ---
    stage_seconds: dict[str, float] = {"query_analyzer": 0.0, "page_summary": 0.0, "visual_chunk": 0.0}

    def _wrap(name: str, bucket: str):
        original = getattr(retrieval_mod, name)

        def wrapped(*a, **kw):
            t0 = time.monotonic()
            try:
                return original(*a, **kw)
            finally:
                stage_seconds[bucket] += time.monotonic() - t0

        setattr(retrieval_mod, name, wrapped)

    _wrap("analyze_query", "query_analyzer")
    _wrap("ensure_page_summaries", "page_summary")
    _wrap("get_or_build_summary", "page_summary")  # retrieval.py가 pending_vlm 승격 시 직접 호출하는 별도 경로
    _wrap("ensure_visual_chunks", "visual_chunk")

    question = args.question.strip()
    run_id = new_run_id(question)
    run_metric = RunMetrics(mode=args.mode, depth="answer")

    error: str | None = None
    retrieval = None
    answer_result: dict = {"final_answer": "", "raw_evidence": [], "images_used": []}
    page_evidence: list = []
    verification: dict = {}

    t_start = time.monotonic()
    t0 = t1 = t2 = t3 = t4 = t_start
    try:
        with run_metrics(run_metric):
            t0 = time.monotonic()
            retrieval = retrieval_mod.run_retrieval(question, config, backend, mode=args.mode)
            t1 = time.monotonic()
            answer_result = generate_answer(question, retrieval, backend, config)
            t2 = time.monotonic()
            page_evidence = build_page_evidence(
                answer_result.get("raw_evidence", []), answer_result.get("images_used", []), retrieval.selected_documents, run_id, config
            )
            t3 = time.monotonic()
            verification = verify_answer(question, answer_result["final_answer"], page_evidence, backend, config)
            t4 = time.monotonic()
    except Exception as e:  # 비교 실험 오케스트레이터가 죽지 않도록 예외를 결과에 담아 흡수
        logger.exception("실행 실패: %s", e)
        error = f"{type(e).__name__}: {e}"
        t4 = time.monotonic()
        if t1 == t_start:
            t1 = t4
        if t2 == t_start:
            t2 = t4
        if t3 == t_start:
            t3 = t4

    total = time.monotonic() - t_start
    retrieval_total = t1 - t0
    retrieval_search = max(0.0, retrieval_total - stage_seconds["query_analyzer"] - stage_seconds["page_summary"] - stage_seconds["visual_chunk"])

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

    result = {
        "pipeline": "test1",
        "mode": args.mode,
        "qid": args.qid,
        "question": question,
        "run_id": run_id,
        "error": error,
        "final_answer": answer_text,
        "abstained": bool(answer_result.get("skip_reason")),
        "skip_reason": answer_result.get("skip_reason"),
        "selected_documents": selected_doc_names,
        "selected_pages": selected_page_numbers,
        "doc_match": (any(d in selected_doc_names for d in expected_documents) if expected_documents else None),
        "page_match": (any(p in selected_page_numbers for p in expected_pages) if expected_pages else None),
        "keyword_recall": (round(len(kw_found) / len(expected_keywords), 3) if expected_keywords else None),
        "keywords_found": kw_found,
        "keywords_missing": kw_missing,
        "is_answer_supported": verification.get("is_answer_supported"),
        "unsupported_claims": verification.get("unsupported_claims", []),
        "stage_timings_seconds": {
            "query_analyzer": round(stage_seconds["query_analyzer"], 3),
            "page_summary": round(stage_seconds["page_summary"], 3),
            "visual_chunk": round(stage_seconds["visual_chunk"], 3),
            "retrieval_search": round(retrieval_search, 3),
            "retrieval_total": round(retrieval_total, 3),
            "answer": round(t2 - t1, 3),
            "evidence": round(t3 - t2, 3),
            "verify": round(t4 - t3, 3),
            "total": round(total, 3),
        },
        "model_calls": {
            "query_analyzer_calls": metrics_dict["query_analyzer_calls"],
            "summary_calls": metrics_dict["summary_calls"],
            "chunk_calls": metrics_dict["chunk_calls"],
            "answer_vlm_calls": metrics_dict["answer_vlm_calls"],
            "verify_llm_calls": metrics_dict["verify_llm_calls"],
            "embed_calls": metrics_dict["embed_calls"],
            "llm_calls_total": metrics_dict["query_analyzer_calls"] + metrics_dict["verify_llm_calls"],
            "vlm_calls_total": metrics_dict["summary_calls"] + metrics_dict["chunk_calls"] + metrics_dict["answer_vlm_calls"],
        },
        "vlm_pages_required_cache_independent": {
            "summary_pages_required": metrics_dict["summary_pages_required"],
            "chunk_pages_required": metrics_dict["chunk_pages_required"],
        },
        "evidence": [
            {
                "image_index": e["image_index"],
                "document_name": e["document_name"],
                "page_number": e["page_number"],
                "confidence": e["confidence"],
                "why_relevant": e["why_relevant"],
                "bbox": e["bbox"],
                "crop_image_path": e["crop_image_path"],
                "highlighted_page_path": e["highlighted_page_path"],
            }
            for e in page_evidence
        ],
    }

    out_path = Path(args.result_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({"qid": args.qid, "mode": args.mode, "total_seconds": result["stage_timings_seconds"]["total"], "error": error}, ensure_ascii=False))


if __name__ == "__main__":
    main()

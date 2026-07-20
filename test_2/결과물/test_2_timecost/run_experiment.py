"""catalog vs no_catalog 비교 실험 오케스트레이터.

test_1차 REPORT.md §10 "카탈로그 실효성 비교 실험"을 test_2(rag2) 파이프라인에 대해
재현한다. rag2 패키지(test_2/rag2)는 전혀 수정하지 않고 import만 하며, rag2가 이미
ingest해 둔 기존 인덱스/캐시(test_2/rag2/index, cache)를 그대로 읽어서 쓴다.

산출물은 이 폴더(test_2_timecost/results/)에만 쓴다 — rag2.evaluate.save_evaluation 같은
rag2의 저장 함수는 호출하지 않으므로 test_2/ 아래에는 어떤 파일도 새로 생기지 않는다.

실행 (test_2와 동일한 conda 환경 필요):
    cmd /c conda activate intern_chatbot && python run_experiment.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent
_TEST2_DIR = _HERE.parent / "test_2"
sys.path.insert(0, str(_TEST2_DIR))

from rag2.answer import generate_answer  # noqa: E402
from rag2.config import Config, load_config  # noqa: E402
from rag2.evaluate import _avg, _keyword_hits  # noqa: E402
from rag2.metrics import RunMetrics, record_timing, run_metrics  # noqa: E402
from rag2.models import Backend, get_backend  # noqa: E402
from rag2.retrieve import RetrievalResult, run_retrieval  # noqa: E402

from retrieve_no_catalog import retrieve_no_catalog  # noqa: E402

logger = logging.getLogger("catalog_ablation")

_DEFAULT_EVAL_FILE = _TEST2_DIR / "rag_eval_dataset.json"

RetrievalFn = Callable[[str, Config, Backend], RetrievalResult]

MODES: dict[str, RetrievalFn] = {
    "catalog": run_retrieval,
    "no_catalog": retrieve_no_catalog,
}


def evaluate_item(item: dict[str, Any], config: Config, backend: Backend, retrieval_fn: RetrievalFn) -> dict[str, Any]:
    """rag2.evaluate.evaluate_item과 동일한 계산식. retrieval_fn만 모드별로 바꿔 끼운다."""
    question = item["question"]
    run_metric = RunMetrics()

    t0 = time.monotonic()
    with run_metrics(run_metric):
        retrieval = retrieval_fn(question, config, backend)
        t1 = time.monotonic()
        record_timing("retrieve", t1 - t0)

        answer_result = generate_answer(question, retrieval, backend, config)
        t2 = time.monotonic()
        record_timing("answer", t2 - t1)
        record_timing("total", t2 - t0)

    metrics_dict = run_metric.to_dict()

    selected_doc_names = [d["document_name"] for d in retrieval.selected_documents]
    selected_page_numbers = [p["page_number"] for p in retrieval.selected_pages]

    result: dict[str, Any] = {
        "id": item["id"],
        "category": item.get("category", ""),
        "question_type": item.get("question_type", ""),
        "answer_path": answer_result["answer_path"],
        "elapsed_seconds": metrics_dict["timings_seconds"].get("total", 0.0),
        "model_calls": metrics_dict["total_model_calls"],
        "selected_documents": selected_doc_names,
        "selected_pages": selected_page_numbers,
        "final_answer": answer_result["final_answer"],
    }

    if item.get("question_type") == "irrelevant":
        result["correctly_rejected"] = answer_result["answer_path"] == "none"
        return result

    expected_documents = item.get("expected_documents", [])
    expected_pages = item.get("expected_pages", [])
    expected_keywords = item.get("expected_answer_keywords", [])
    found_kw, missing_kw = _keyword_hits(answer_result["final_answer"], expected_keywords)

    result["doc_match"] = any(d in selected_doc_names for d in expected_documents)
    result["page_match"] = any(p in selected_page_numbers for p in expected_pages)
    result["keyword_hit_rate"] = round(len(found_kw) / len(expected_keywords), 3) if expected_keywords else None
    result["missing_keywords"] = missing_kw
    return result


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """rag2.evaluate.run_evaluation의 집계식과 동일."""
    ok = [r for r in results if "error" not in r]
    relevant = [r for r in ok if "doc_match" in r]
    irrelevant = [r for r in ok if "correctly_rejected" in r]
    kw_rates = [r["keyword_hit_rate"] for r in relevant if r["keyword_hit_rate"] is not None]

    return {
        "total_items": len(ok),
        "error_items": len(results) - len(ok),
        "relevant_items": len(relevant),
        "irrelevant_items": len(irrelevant),
        "doc_hit_rate": _avg([float(r["doc_match"]) for r in relevant]),
        "page_hit_rate": _avg([float(r["page_match"]) for r in relevant]),
        "avg_keyword_hit_rate": _avg(kw_rates),
        "irrelevant_correctly_rejected_rate": _avg([float(r["correctly_rejected"]) for r in irrelevant]),
        "answer_path_counts": {
            path: sum(1 for r in ok if r["answer_path"] == path) for path in ("text", "vision", "none")
        },
        "avg_elapsed_seconds": _avg([r["elapsed_seconds"] for r in ok]),
        "total_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in ok), 2),
        "avg_model_calls": _avg([float(r["model_calls"]) for r in ok]),
        "avg_selected_doc_count": _avg([float(len(r["selected_documents"])) for r in ok]),
        "avg_selected_page_count": _avg([float(len(r["selected_pages"])) for r in ok]),
    }


def run_mode(mode: str, eval_items: list[dict[str, Any]], config: Config, backend: Backend) -> dict[str, Any]:
    retrieval_fn = MODES[mode]
    results = []
    for i, item in enumerate(eval_items, start=1):
        logger.info("[%s] %d/%d: %s", mode, i, len(eval_items), item["id"])
        try:
            results.append(evaluate_item(item, config, backend, retrieval_fn))
        except Exception as e:
            logger.exception("[%s] %s 실패: %s", mode, item["id"], e)
            results.append({"id": item["id"], "error": str(e)})
    return {"items": results, "aggregate": aggregate_results(results)}


def render_report_markdown(compare: dict[str, Any]) -> str:
    modes = compare["modes"]
    lines = [
        "# test_2 카탈로그 유무 비교 실험 (catalog vs no_catalog)",
        "",
        f"- 생성 시각: {compare['generated_at']}",
        f"- 평가 문항 수: {compare['eval_item_count']} (`{_DEFAULT_EVAL_FILE.name}`, core 13 + irrelevant 3)",
        f"- 대상 파이프라인: `test_2/rag2` (수정 없이 import), 인덱스/캐시는 기존 ingest 결과 재사용",
        "",
        "## 요약 비교표",
        "",
    ]

    cols = [
        ("total_items", "문항"),
        ("doc_hit_rate", "doc_hit"),
        ("page_hit_rate", "page_hit"),
        ("avg_keyword_hit_rate", "kw_hit"),
        ("irrelevant_correctly_rejected_rate", "무관거절"),
        ("answer_path_counts", "경로(text/vision/none)"),
        ("avg_selected_doc_count", "avg_docs"),
        ("avg_model_calls", "avg_모델호출"),
        ("avg_elapsed_seconds", "avg_초"),
        ("total_elapsed_seconds", "총_초"),
    ]

    header = "| mode | " + " | ".join(label for _, label in cols) + " |"
    sep = "|---|" + "---|" * len(cols)
    lines.append(header)
    lines.append(sep)
    for mode in modes:
        agg = compare["per_mode"][mode]["aggregate"]
        cells = []
        for key, _ in cols:
            v = agg.get(key)
            if key == "answer_path_counts" and isinstance(v, dict):
                cells.append(f"{v.get('text', 0)}/{v.get('vision', 0)}/{v.get('none', 0)}")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v) if v is not None else "-")
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 문항별 상세 (doc_match / page_match / answer_path / 소요초)")
    lines.append("")
    header2 = "| id | " + " | ".join(f"{m}.doc" for m in modes) + " | " + " | ".join(f"{m}.page" for m in modes) + " | " + " | ".join(f"{m}.path" for m in modes) + " | " + " | ".join(f"{m}.sec" for m in modes) + " |"
    lines.append(header2)
    lines.append("|---|" + "---|" * (len(modes) * 4))
    by_id: dict[str, dict[str, dict]] = {}
    for mode in modes:
        for r in compare["per_mode"][mode]["items"]:
            by_id.setdefault(r["id"], {})[mode] = r
    for qid in by_id:
        row = [qid]
        for mode in modes:
            r = by_id[qid].get(mode, {})
            row.append("-" if "doc_match" not in r else str(r["doc_match"]))
        for mode in modes:
            r = by_id[qid].get(mode, {})
            row.append("-" if "page_match" not in r else str(r["page_match"]))
        for mode in modes:
            r = by_id[qid].get(mode, {})
            row.append(r.get("answer_path", r.get("error", "-")))
        for mode in modes:
            r = by_id[qid].get(mode, {})
            sec = r.get("elapsed_seconds")
            row.append(f"{sec:.1f}" if isinstance(sec, (int, float)) else "-")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = load_config()  # rag2 패키지 내장 config.yaml 사용 (경로는 test_2/rag2 기준 그대로)
    backend = get_backend(config)

    with open(_DEFAULT_EVAL_FILE, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    logger.info("평가 문항 %d개, 모드 %s", len(eval_items), list(MODES))

    per_mode = {}
    for mode in MODES:
        logger.info("=== 모드 시작: %s ===", mode)
        per_mode[mode] = run_mode(mode, eval_items, config, backend)

    compare = {
        "modes": list(MODES),
        "eval_item_count": len(eval_items),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "per_mode": per_mode,
    }

    results_dir = _HERE / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = results_dir / f"compare_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)

    report_md = render_report_markdown(compare)
    (results_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

    print(f"\n[저장됨] {json_path}")
    print(f"[저장됨] {results_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()

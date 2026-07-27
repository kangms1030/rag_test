"""rag_eval_dataset.json으로 파이프라인 전체를 일괄 평가.

`ask`(단일 질문용)와 달리 문항마다 전체 answer_*.json을 남기지 않는다 — 16문항을
반복 실행하면 그 산출물이 그대로 쌓여 outputs/를 어지럽히므로, 문항별 결과는
집계 리포트 안에 압축된 형태로만 담고 최종 리포트 파일 하나만 저장한다.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answer import generate_answer
from .config import Config
from .metrics import RunMetrics, record_timing, run_metrics
from .models import Backend
from .retrieve import run_retrieval

logger = logging.getLogger(__name__)


def _keyword_hits(answer_text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    found = [kw for kw in keywords if kw in answer_text]
    missing = [kw for kw in keywords if kw not in answer_text]
    return found, missing


def evaluate_item(item: dict[str, Any], config: Config, backend: Backend) -> dict[str, Any]:
    question = item["question"]
    run_metric = RunMetrics()

    t0 = time.monotonic()
    with run_metrics(run_metric):
        retrieval = run_retrieval(question, config, backend)
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


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def run_evaluation(eval_items: list[dict[str, Any]], config: Config, backend: Backend) -> dict[str, Any]:
    results = []
    for i, item in enumerate(eval_items, start=1):
        logger.info("평가 %d/%d: %s", i, len(eval_items), item["id"])
        results.append(evaluate_item(item, config, backend))

    relevant = [r for r in results if "doc_match" in r]
    irrelevant = [r for r in results if "correctly_rejected" in r]
    kw_rates = [r["keyword_hit_rate"] for r in relevant if r["keyword_hit_rate"] is not None]

    aggregate = {
        "total_items": len(results),
        "relevant_items": len(relevant),
        "irrelevant_items": len(irrelevant),
        "doc_hit_rate": _avg([float(r["doc_match"]) for r in relevant]),
        "page_hit_rate": _avg([float(r["page_match"]) for r in relevant]),
        "avg_keyword_hit_rate": _avg(kw_rates),
        "irrelevant_correctly_rejected_rate": _avg([float(r["correctly_rejected"]) for r in irrelevant]),
        "answer_path_counts": {
            path: sum(1 for r in results if r["answer_path"] == path) for path in ("text", "vision", "none")
        },
        "avg_elapsed_seconds": _avg([r["elapsed_seconds"] for r in results]),
        "total_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in results), 2),
        "avg_model_calls": _avg([float(r["model_calls"]) for r in results]),
    }

    return {"items": results, "aggregate": aggregate}


def save_evaluation(result: dict[str, Any], config: Config) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"evaluation_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("평가 리포트 저장: %s", path)
    return path

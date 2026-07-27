"""최종 답변 JSON 조립 + 저장."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_NO_DOC_ANSWER = "선택된 문서에서 확인 불가"


def _config_snapshot(config: Config) -> dict[str, Any]:
    return {
        "top_docs": config.top_docs,
        "top_pages": config.top_pages,
        "min_dense_similarity": config.min_dense_similarity,
        "min_doc_score": config.min_doc_score,
        "doc_score_gap_ratio": config.doc_score_gap_ratio,
        "parser": config.parser,
        "text_answer_model": config.text_answer_model,
        "vision_answer_model": config.vision_answer_model,
        "embedding_model": config.embedding_model,
    }


def build_final_json(
    question: str,
    *,
    run_id: str,
    config: Config,
    selected_documents: list[dict[str, Any]],
    selected_pages: list[dict[str, Any]],
    answer_path: str,
    final_answer: str,
    evidence: list[dict[str, Any]],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not selected_documents or not selected_pages:
        run_status = "no_answer"
    else:
        run_status = "success"

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_status": run_status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": _config_snapshot(config),
        "question": question,
        "selected_documents": selected_documents,
        "selected_pages": selected_pages,
        "answer_path": answer_path,  # "text" | "vision"
        "final_answer": final_answer,
        "evidence": evidence,
        "metrics": metrics or {},
    }


def save_final_json(obj: dict[str, Any], config: Config, run_id: str) -> Path:
    category = "success" if obj["run_status"] == "success" else "no_answer"
    out_dir = config.output_dir / "runs" / category
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"answer_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    logger.info("최종 답변 JSON 저장: %s", path)
    return path

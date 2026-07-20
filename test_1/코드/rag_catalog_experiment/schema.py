"""최종 답변 JSON 조립 + 저장 + 최소 스키마 검증."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)

#: 스키마가 하위 호환 없이 바뀔 때마다 올린다. v1(암묵적, 필드 없음): source_images/
#: page_evidence[].image_index/metrics/retrieval_mode 없음. v2: 이 필드들이 전부 존재.
SCHEMA_VERSION = 2


def classify_answer(obj: dict[str, Any]) -> str:
    """answer JSON 하나를 success|failed|mock|old_schema 중 하나로 분류.

    `runs.py`(사후 산출물 정리)와 `save_final_json`(저장 시점 즉시 분류)이 공유하는
    단일 판단 기준 — 두 곳에서 분류 규칙이 따로 놀지 않게 여기 한 곳에 둔다.
    `schema_version`이 없는 산출물(과거 산출물)은 mock이 아니면 전부 old_schema로 보낸다.
    """
    if obj.get("mock") is True or obj.get("run_status") == "mock":
        return "mock"
    if "schema_version" not in obj:
        return "old_schema"
    status = obj.get("run_status")
    if status in ("success", "no_answer"):
        return "success"
    if status == "failed":
        return "failed"
    return "old_schema"


def _config_snapshot(config: Config) -> dict[str, Any]:
    """비교 실험 재현에 필요한 핵심 설정값만 남긴다 (전체 Config는 경로 등 잡음이 많음)."""
    return {
        "top_docs": config.top_docs,
        "top_pages": config.top_pages,
        "top_chunks": config.top_chunks,
        "page_prefilter_topn": config.page_prefilter_topn,
        "min_dense_similarity": config.min_dense_similarity,
        "min_doc_score": config.min_doc_score,
        "doc_score_gap_ratio": config.doc_score_gap_ratio,
        "llm_model": config.llm_model,
        "vlm_model": config.vlm_model,
        "embedding_model": config.embedding_model,
    }


def build_final_json(
    question: str,
    retrieval: RetrievalResult,
    answer_result: dict[str, Any],
    page_evidence: list[dict[str, Any]],
    verification: dict[str, Any],
    *,
    run_id: str,
    retrieval_mode: str,
    config: Config,
    metrics: dict[str, Any] | None = None,
    mock: bool = False,
    run_status: str | None = None,
) -> dict[str, Any]:
    if run_status is None:
        if mock:
            run_status = "mock"
        elif not retrieval.selected_documents or not retrieval.selected_pages:
            run_status = "no_answer"
        else:
            run_status = "success"

    qa = retrieval.query_analysis
    obj = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_status": run_status,
        "retrieval_mode": retrieval_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": _config_snapshot(config),
        "question": question,
        "query_analysis": {
            "intent": qa.get("intent", ""),
            "keywords": qa.get("keywords", []),
            "domain_terms": qa.get("domain_terms", []),
            "required_data_type": qa.get("required_data_type", ""),
            "retrieval_query": qa.get("retrieval_query", ""),
            "reason": qa.get("reason", ""),
        },
        "selected_documents": [
            {
                "rank": d["rank"],
                "document_name": d["document_name"],
                "file_path": d["file_path"],
                "catalog_row_id": d["catalog_row_id"],
                "selection_score": d["selection_score"],
                "selection_reason": d["selection_reason"],
            }
            for d in retrieval.selected_documents
        ],
        "selected_pages": [
            {
                "document_name": p["document_name"],
                "page_number": p["page_number"],
                "page_score": p["page_score"],
                "page_summary": p["page_summary"],
                "image_path": p["image_path"],
            }
            for p in retrieval.selected_pages
        ],
        "selected_visual_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "chunk_type": c["chunk_type"],
                "summary": c["summary"],
                "bbox": c["bbox"],
                "score": c["score"],
                "why_relevant": c["why_relevant"],
            }
            for c in retrieval.selected_chunks
        ],
        # final_answer 본문의 "(이미지 N)" 표현을 실제 문서/페이지로 되돌리기 위한 매니페스트.
        # VLM에게 보여준 이미지 목록과 순번이 정확히 일치한다.
        "source_images": [
            {
                "image_index": i,
                "document_name": img["document_name"],
                "page_number": img["page_number"],
                "image_path": img["image_path"],
            }
            for i, img in enumerate(answer_result.get("images_used", []), start=1)
        ],
        "page_evidence": page_evidence,
        "final_answer": answer_result["final_answer"],
        "verification": verification,
        "metrics": metrics or {},
    }
    if mock:
        obj["mock"] = True
    return obj


class SchemaValidationError(ValueError):
    pass


def validate_final_json(obj: dict[str, Any]) -> None:
    required_top = [
        "schema_version",
        "run_id",
        "run_status",
        "retrieval_mode",
        "question",
        "query_analysis",
        "selected_documents",
        "selected_pages",
        "selected_visual_chunks",
        "source_images",
        "page_evidence",
        "final_answer",
        "verification",
        "metrics",
    ]
    missing = [k for k in required_top if k not in obj]
    if missing:
        raise SchemaValidationError(f"최상위 필드 누락: {missing}")

    for ev in obj["page_evidence"]:
        if not isinstance(ev.get("image_index"), int):
            raise SchemaValidationError(f"page_evidence.image_index 누락 또는 정수 아님: {ev}")
        bbox = ev.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
        if None in (x1, y1, x2, y2):
            raise SchemaValidationError(f"page_evidence bbox 필드 누락: {ev}")
        if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
            raise SchemaValidationError(f"page_evidence bbox가 [0,1] 범위를 벗어남: {bbox}")
        if x2 <= x1 or y2 <= y1:
            raise SchemaValidationError(f"page_evidence bbox가 x2>x1, y2>y1을 만족하지 않음: {bbox}")
        for path_key in ("crop_image_path", "highlighted_page_path"):
            p = Path(ev[path_key])
            if not p.exists():
                raise SchemaValidationError(f"{path_key} 파일이 존재하지 않음: {p}")

    for chunk in obj["selected_visual_chunks"]:
        bbox = chunk.get("bbox", {})
        x1, y1, x2, y2 = bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")
        if None in (x1, y1, x2, y2) or x2 <= x1 or y2 <= y1:
            raise SchemaValidationError(f"selected_visual_chunks bbox 유효성 실패: {chunk}")


def save_final_json(obj: dict[str, Any], config: Config, run_id: str) -> Path:
    """분류 폴더(outputs/runs/{success,failed,mock}/)에 바로 저장한다.

    저장 시점에 이미 run_status를 알고 있으므로 `clean-runs`로 사후 이동할 필요가 없다.
    `clean-runs`는 이 스키마 이전(schema_version 없음)에 만들어진 과거 산출물 정리용으로 남는다.
    """
    category = classify_answer(obj)
    out_dir = config.output_dir / "runs" / category
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"answer_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    logger.info("최종 답변 JSON 저장: %s", path)
    return path

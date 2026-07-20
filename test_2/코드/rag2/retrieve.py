"""질의 임베딩(1회) -> catalog_index 게이트+top_docs -> page_index(문서 필터) 검색 ->
근거 페이지 메타 기반 결정론적 경로 판정(모델 호출 없음).

test_1차의 query analyzer/lazy VLM 요약을 제거했다 — 문서/페이지 선정은 원문 질문
그대로 BM25+dense RRF에 맡기고, ingest 때 전부 선연산된 page_index만 읽는다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from . import metrics
from .config import Config
from .index import HybridIndex, ScoredItem, get_index
from .models import Backend

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    question: str
    selected_documents: list[dict[str, Any]]
    selected_pages: list[dict[str, Any]]
    #: "text" | "vision" | "none"(문서/페이지를 못 찾음)
    answer_path: str = "none"
    route_reason: str = ""


def _select_documents(
    question: str, index: HybridIndex, config: Config, query_embedding: list[float]
) -> list[dict[str, Any]]:
    results = index.query(question, n_results=max(config.top_docs, 5), query_embedding=query_embedding)
    if not results:
        return []

    top = results[0]
    dense_sim = top.dense_similarity or 0.0
    if dense_sim < config.min_dense_similarity:
        logger.info("카탈로그: 최상위 후보 dense 유사도 부족 (%.4f < %.2f) -> 후보 문서 없음", dense_sim, config.min_dense_similarity)
        return []
    if top.score < config.min_doc_score:
        logger.info("카탈로그: 최상위 점수(%.4f) < min_doc_score(%.4f) -> 후보 문서 없음", top.score, config.min_doc_score)
        return []

    picked = [results[0]]
    if len(results) > 1 and results[1].score >= config.doc_score_gap_ratio * top.score:
        picked.append(results[1])
    picked = picked[: config.top_docs]

    selected = []
    for rank, item in enumerate(picked, start=1):
        m = item.metadata
        selected.append(
            {
                "rank": rank,
                "document_name": m["document_name"],
                "file_path": m["file_path"],
                "doc_slug": m["doc_slug"],
                "catalog_row_id": item.id,
                "selection_score": round(item.score, 6),
                "selection_reason": f"카탈로그 하이브리드 검색 상위 {rank}위 (dense_rank={item.dense_rank}, bm25_rank={item.bm25_rank})",
            }
        )
    return selected


def _select_pages(
    question: str,
    selected_documents: list[dict[str, Any]],
    page_index: HybridIndex,
    config: Config,
    query_embedding: list[float],
) -> list[ScoredItem]:
    per_doc: list[list[ScoredItem]] = []
    for doc in selected_documents:
        results = page_index.query(
            question, n_results=config.top_pages, where={"doc_slug": doc["doc_slug"]}, query_embedding=query_embedding
        )
        per_doc.append(results)
    merged = sorted([r for lst in per_doc for r in lst], key=lambda x: -x.score)
    return merged[: config.top_pages]


def _route(top_pages: list[ScoredItem], config: Config) -> tuple[str, str]:
    """1순위 근거 페이지의 메타데이터만 보고 결정론적으로 경로를 정한다(모델 호출 없음)."""
    if not top_pages:
        return "none", "근거 페이지 없음"

    best = top_pages[0].metadata
    if best.get("page_type") == "figure" and float(best.get("figure_area_ratio", 0.0)) >= config.figure_area_ratio_threshold:
        return "vision", f"1순위 페이지가 그림 위주(figure_area_ratio={best.get('figure_area_ratio'):.2f})"
    if config.scanned_table_verify and bool(best.get("is_scanned")) and bool(best.get("has_table")):
        return "vision", "1순위 페이지가 스캔+표 -> 숫자 교차확인을 위해 비전 경로"
    return "text", "1순위 페이지가 텍스트/구조화 표 -> 텍스트 경로(기본)"


def run_retrieval(question: str, config: Config, backend: Backend) -> RetrievalResult:
    catalog_index = get_index("catalog_index", config, backend)
    page_index = get_index("page_index", config, backend)

    query_embedding = backend.embed([question], is_query=True)[0]
    metrics.record_embed()

    selected_documents = _select_documents(question, catalog_index, config, query_embedding)
    if not selected_documents:
        return RetrievalResult(question, [], [], answer_path="none", route_reason="카탈로그에서 관련 문서를 찾지 못함")

    top_page_results = _select_pages(question, selected_documents, page_index, config, query_embedding)
    if not top_page_results:
        return RetrievalResult(question, selected_documents, [], answer_path="none", route_reason="선정 문서에서 관련 페이지를 찾지 못함")

    selected_pages = [
        {
            "document_name": r.metadata["document_name"],
            "page_number": r.metadata["page_number"],
            "page_score": round(r.score, 6),
            "page_type": r.metadata.get("page_type", "text"),
            "is_scanned": bool(r.metadata.get("is_scanned")),
            "has_table": bool(r.metadata.get("has_table")),
            "table_markdown": r.metadata.get("table_markdown", ""),
            "table_crop_path": r.metadata.get("table_crop_path", ""),
            "page_image_path": r.metadata["page_image_path"],
            "text": r.text,
        }
        for r in top_page_results
    ]

    answer_path, route_reason = _route(top_page_results, config)
    return RetrievalResult(question, selected_documents, selected_pages, answer_path=answer_path, route_reason=route_reason)

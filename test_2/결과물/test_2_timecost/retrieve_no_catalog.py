"""카탈로그 없이 page_index를 문서 범위 제한 없이 전역 검색하는 baseline.

test_1차 REPORT.md §10의 `no_catalog` 모드를 test_2(rag2) 파이프라인에 이식한 것이다.
rag2.retrieve.run_retrieval과 차이는 딱 한 곳: 문서를 먼저 선정(catalog_index 게이트)한 뒤
그 안에서만 페이지를 찾는 대신, page_index를 처음부터 전역(where=None)으로 검색하고
상위 페이지들의 doc_slug를 역산해 selected_documents를 만든다.

이후 단계(라우팅 `_route`, 답변 생성 `generate_answer`, 지표 계산)는 rag2 코드를
그대로 재사용한다 — 이 파일은 rag2를 전혀 수정하지 않고 import만 한다.
"""
from __future__ import annotations

import logging
from typing import Any

from rag2 import metrics
from rag2.config import Config
from rag2.index import get_index
from rag2.models import Backend
from rag2.retrieve import RetrievalResult, _route

logger = logging.getLogger(__name__)


def retrieve_no_catalog(question: str, config: Config, backend: Backend) -> RetrievalResult:
    page_index = get_index("page_index", config, backend)

    query_embedding = backend.embed([question], is_query=True)[0]
    metrics.record_embed()

    # 문서 필터 없이 전역 검색 (where=None) — catalog_index 게이트를 건너뛴다.
    top_page_results = page_index.query(question, n_results=config.top_pages, query_embedding=query_embedding)

    if not top_page_results:
        return RetrievalResult(question, [], [], answer_path="none", route_reason="page_index에 결과 없음")

    # test_1차 no_catalog와 동일한 무관 질문 거절 게이트: 최상위 페이지의 절대 dense 유사도.
    top_dense_sim = top_page_results[0].dense_similarity or 0.0
    if top_dense_sim < config.min_dense_similarity:
        logger.info(
            "no_catalog: 최상위 페이지 dense 유사도 부족 (%.4f < %.2f) -> 후보 없음",
            top_dense_sim,
            config.min_dense_similarity,
        )
        return RetrievalResult(question, [], [], answer_path="none", route_reason="전역 검색 결과의 관련성 부족(dense gate)")

    # 상위 페이지들의 doc_slug를 순위 순서로 역산해 selected_documents를 구성.
    selected_documents: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    for r in top_page_results:
        m = r.metadata
        slug = m["doc_slug"]
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)
        selected_documents.append(
            {
                "rank": len(selected_documents) + 1,
                "document_name": m["document_name"],
                "file_path": m["file_path"],
                "doc_slug": slug,
                "catalog_row_id": None,
                "selection_score": round(r.score, 6),
                "selection_reason": "no_catalog: page_index 전역검색 상위 페이지에서 문서 역산",
            }
        )

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

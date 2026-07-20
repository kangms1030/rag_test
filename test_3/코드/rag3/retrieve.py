"""검색 v2 (Phase 1): 청크 전역 하이브리드 검색 -> 리랭크 -> small-to-big 페이지 승격 -> 라우팅 v2.

test_2 대비 변경(Phase 0 근거):
- 카탈로그 게이트 제거(기본). 무관 질문 거절은 리랭크 절대점수 floor로 대체. (게이트가 소규모에서 손해)
- 페이지 통짜 대신 청크로 검색해 정답을 상위로 끌어올림(리랭커) -> "정답 2~3순위" 실패 공략.
- 라우팅은 단일 1순위가 아니라 승격된 페이지 집합 기준. "스캔+표->vision" 규칙 폐기(text-first).
use_catalog_gate=true면 test_2의 게이트 경로(_run_retrieval_gated)로 폴백한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from . import metrics
from .config import Config
from .flat_index import get_flat_chunk_index
from .index import HybridIndex, get_index
from .models import Backend
from .page_store import fetch_pages
from .rerank import get_reranker

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    question: str
    selected_documents: list[dict[str, Any]]
    selected_pages: list[dict[str, Any]]
    answer_path: str = "none"       # "text" | "vision" | "none"
    route_reason: str = ""
    rerank_top_score: float | None = None
    candidate_chunks: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Phase 1 기본 경로: 청크 검색 + 리랭커 + small-to-big
# ---------------------------------------------------------------------------

def _route_v2(pages: list[dict[str, Any]], config: Config) -> tuple[str, str]:
    """승격된 페이지 집합 기준 결정론적 경로. text가 기본(숫자 정확도 100%).

    vision은 1순위 근거가 '텍스트 빈약한 그림 페이지'일 때만(도표 라벨은 텍스트로 안 뽑히므로).
    스캔+표는 text로 보낸다 — MinerU OCR/HTML 텍스트가 VLM 전사보다 정확(Phase 0-A/fact §4).
    """
    if not pages:
        return "none", "근거 페이지 없음"
    best = pages[0]
    fr = float(best.get("figure_area_ratio", 0.0))
    if best.get("page_type") == "figure" and fr >= config.figure_area_ratio_threshold:
        return "vision", f"1순위 근거가 그림 위주 페이지(figure_area_ratio={fr:.2f})"
    return "text", "1순위 근거에 사용 가능한 텍스트/표 존재 -> 텍스트 경로(기본)"


def _rerank_text(indexed_text: str, config: Config) -> str:
    """리랭커 입력 텍스트. rerank_use_raw_text면 ingest가 주입한 프리픽스 라인을 제거하고
    원문 청크만 남긴다(포맷: "{prefix}...{head} | p{n}\\n{원문}" — 첫 개행 이후가 원문).

    카탈로그 프리픽스(문서/분류/범위/키워드)는 dense 임베딩의 형제문서 변별엔 유용하나
    cross-encoder 리랭커에는 전 청크 공통 boilerplate라 질문-청크 관련도를 희석한다.
    """
    if getattr(config, "rerank_use_raw_text", False):
        nl = indexed_text.find("\n")
        if nl != -1:
            return indexed_text[nl + 1:]
    return indexed_text


def _agg_page_score(scores: list[float], config: Config) -> float:
    """청크 점수들을 페이지 점수로 집계. max(기본) 또는 sum_topk(best + decay*나머지 top-k)."""
    if getattr(config, "page_score_agg", "max") == "sum_topk" and len(scores) > 1:
        top = sorted(scores, reverse=True)[: getattr(config, "page_score_topk", 3)]
        return top[0] + getattr(config, "page_score_decay", 0.5) * sum(top[1:])
    return max(scores)


def _run_retrieval_reranked(question: str, config: Config, backend: Backend) -> RetrievalResult:
    chunk_index = get_flat_chunk_index(config, backend)

    q_emb = backend.embed([question], is_query=True)[0]
    metrics.record_embed()

    cands = chunk_index.query(question, n_results=config.retrieve_candidates, query_embedding=q_emb)
    if not cands:
        return RetrievalResult(question, [], [], answer_path="none", route_reason="청크 검색 결과 없음")

    reranker = get_reranker(config)
    hits = reranker.rank(question, [_rerank_text(c.text, config) for c in cands])
    metrics.record_rerank()
    top_score = hits[0].score if hits else float("-inf")

    cand_dump = [
        {"chunk_id": cands[h.index].metadata.get("chunk_id"),
         "doc_slug": cands[h.index].metadata.get("doc_slug"),
         "page_number": cands[h.index].metadata.get("page_number"),
         "block_type": cands[h.index].metadata.get("block_type"),
         "rerank_score": round(h.score, 4)}
        for h in hits[: config.retrieve_candidates]
    ]

    if top_score < config.rerank_score_floor:
        return RetrievalResult(
            question, [], [], answer_path="none",
            route_reason=f"리랭크 최고점수 {top_score:.3f} < floor {config.rerank_score_floor} -> 무관/근거없음",
            rerank_top_score=round(top_score, 4), candidate_chunks=cand_dump,
        )

    # small-to-big: 청크를 페이지로 집계. 페이지 점수는 _agg_page_score(max 또는 sum_topk).
    page_best: dict[tuple, dict] = {}
    for h in hits:
        c = cands[h.index]
        m = c.metadata
        key = (m.get("doc_slug"), m.get("page_number"))
        if key not in page_best:
            page_best[key] = {"scores": [h.score], "meta": m, "chunks": [c]}
        else:
            page_best[key]["scores"].append(h.score)
            page_best[key]["chunks"].append(c)
    for p in page_best.values():
        p["score"] = _agg_page_score(p["scores"], config)
    ordered = sorted(page_best.values(), key=lambda p: -p["score"])[: config.final_pages]

    page_ids = [f'{p["meta"]["doc_slug"]}_p{int(p["meta"]["page_number"]):04d}' for p in ordered]
    id2text, id2meta = fetch_pages(config, page_ids)

    selected_pages: list[dict[str, Any]] = []
    for pid, p in zip(page_ids, ordered):
        pm = id2meta.get(pid, p["meta"])
        selected_pages.append({
            "document_name": pm.get("document_name", p["meta"].get("document_name")),
            "page_number": int(pm.get("page_number", p["meta"].get("page_number"))),
            "page_score": round(p["score"], 4),
            "page_type": pm.get("page_type", "text"),
            "is_scanned": bool(pm.get("is_scanned")),
            "has_table": bool(pm.get("has_table")),
            "figure_area_ratio": float(pm.get("figure_area_ratio", 0.0)),
            "table_markdown": pm.get("table_markdown", ""),
            "table_crop_path": pm.get("table_crop_path", ""),
            "page_image_path": pm.get("page_image_path", ""),
            "text": id2text.get(pid, ""),
            "matched_chunks": [c.metadata.get("chunk_id") for c in p["chunks"][:3]],
        })

    # 페이지에서 문서 집합 도출(게이트 대체 — doc_hit 평가/근거표시용)
    seen: dict[str, dict] = {}
    for sp in selected_pages:
        dn = sp["document_name"]
        if dn not in seen:
            seen[dn] = {"rank": len(seen) + 1, "document_name": dn, "selection_score": sp["page_score"]}
    selected_documents = list(seen.values())

    answer_path, reason = _route_v2(selected_pages, config)
    return RetrievalResult(
        question, selected_documents, selected_pages,
        answer_path=answer_path, route_reason=reason,
        rerank_top_score=round(top_score, 4), candidate_chunks=cand_dump,
    )


# ---------------------------------------------------------------------------
# 폴백 경로: test_2 카탈로그 게이트 (use_catalog_gate=true일 때만)
# ---------------------------------------------------------------------------

def _select_documents(question, index, config, query_embedding):
    results = index.query(question, n_results=max(config.top_docs, 5), query_embedding=query_embedding)
    if not results:
        return []
    top = results[0]
    if (top.dense_similarity or 0.0) < config.min_dense_similarity:
        return []
    if top.score < config.min_doc_score:
        return []
    picked = [results[0]]
    if len(results) > 1 and results[1].score >= config.doc_score_gap_ratio * top.score:
        picked.append(results[1])
    picked = picked[: config.top_docs]
    return [
        {"rank": r, "document_name": it.metadata["document_name"], "file_path": it.metadata["file_path"],
         "doc_slug": it.metadata["doc_slug"], "catalog_row_id": it.id, "selection_score": round(it.score, 6)}
        for r, it in enumerate(picked, start=1)
    ]


def _select_pages(question, selected_documents, page_index, config, query_embedding):
    per_doc = []
    for doc in selected_documents:
        per_doc.append(page_index.query(question, n_results=config.top_pages,
                                        where={"doc_slug": doc["doc_slug"]}, query_embedding=query_embedding))
    merged = sorted([r for lst in per_doc for r in lst], key=lambda x: -x.score)
    return merged[: config.top_pages]


def _run_retrieval_gated(question: str, config: Config, backend: Backend) -> RetrievalResult:
    catalog_index = get_index("catalog_index", config, backend)
    page_index = get_index("page_index", config, backend)
    q_emb = backend.embed([question], is_query=True)[0]
    metrics.record_embed()
    docs = _select_documents(question, catalog_index, config, q_emb)
    if not docs:
        return RetrievalResult(question, [], [], answer_path="none", route_reason="카탈로그에서 관련 문서 못 찾음")
    top_pages = _select_pages(question, docs, page_index, config, q_emb)
    if not top_pages:
        return RetrievalResult(question, docs, [], answer_path="none", route_reason="선정 문서에서 관련 페이지 못 찾음")
    selected_pages = [{
        "document_name": r.metadata["document_name"], "page_number": r.metadata["page_number"],
        "page_score": round(r.score, 6), "page_type": r.metadata.get("page_type", "text"),
        "is_scanned": bool(r.metadata.get("is_scanned")), "has_table": bool(r.metadata.get("has_table")),
        "figure_area_ratio": float(r.metadata.get("figure_area_ratio", 0.0)),
        "table_markdown": r.metadata.get("table_markdown", ""), "table_crop_path": r.metadata.get("table_crop_path", ""),
        "page_image_path": r.metadata["page_image_path"], "text": r.text,
    } for r in top_pages]
    answer_path, reason = _route_v2(selected_pages, config)
    return RetrievalResult(question, docs, selected_pages, answer_path=answer_path, route_reason=reason)


def run_retrieval(question: str, config: Config, backend: Backend) -> RetrievalResult:
    if config.use_catalog_gate:
        return _run_retrieval_gated(question, config, backend)
    return _run_retrieval_reranked(question, config, backend)

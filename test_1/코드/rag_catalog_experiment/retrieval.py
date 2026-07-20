"""Query Analyzer + catalog/no_catalog/filename_only 3-모드 retrieval.

세 모드는 "문서를 어떻게 좁히는가"만 다르고 이후 페이지→청크→답변 꼬리는 공유한다.

- catalog (기본): 질문 -> catalog_index로 문서 ≤2개 선정(doc-first) -> 선정 문서마다
  페이지 축소(텍스트 prefilter 또는 스캔 전체) -> VLM 요약(lazy) -> page_index 재검색.
- filename_only: catalog와 동일한 doc-first 구조이나 파일명+폴더명만 색인한
  filename_index로 문서를 선정한다. catalog 설명문 자체의 효과인지 파일명만으로도
  충분한지 분리하기 위한 baseline.
- no_catalog: 문서 선정 단계가 없다(page-first). page_index를 문서 범위 제한 없이
  전역 검색하고, 최종 후보 페이지의 문서를 역산해 selected_documents를 채운다.

카탈로그 방식이 실제로 비용/정확도에 도움이 되는지 비교하려면 이 세 경로의 결과를
동일한 스키마(RetrievalResult)로 맞춰야 한다 — 그래야 answer.py/schema.py가 모드를
몰라도 되고, evaluate.py가 모드별로 같은 지표를 낼 수 있다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from . import metrics
from .config import Config
from .indexes import HybridIndex, ScoredItem, get_index
from .models import Backend, LLMJsonError
from .page_summary import ensure_page_summaries, get_or_build_summary
from .pdf_parse import DocumentInfo, PageRecord, get_or_parse_document
from .tokenizer import tokenize_ko
from .visual_chunk import ensure_visual_chunks

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    catalog = "catalog"
    no_catalog = "no_catalog"
    filename_only = "filename_only"


Depth = Literal["docs", "pages", "answer"]

PROMPT_QUERY_ANALYZER = """다음 사용자 질문을 분석해줘.

질문: {question}

반드시 아래 JSON 형식으로만 답해줘. 다른 텍스트는 쓰지 마.
{{
  "intent": "질문의 핵심 의도를 한 문장으로",
  "keywords": ["핵심 키워드1", "핵심 키워드2"],
  "domain_terms": ["전문용어/제품명/장비명 등"],
  "required_data_type": "text|table|diagram|procedure|unknown 중 하나",
  "candidate_filters": {{}},
  "retrieval_query": "문서 검색에 쓸 정제된 검색어 문장",
  "reason": "이렇게 분석한 이유"
}}

이 질문은 학교 유무선 네트워크(무선 AP, 스쿨넷, 통합관제시스템, MDM 등) 관련 문서에서
답을 찾기 위한 것이다. keywords/domain_terms에는 네트워크·장비·기관명을 우선적으로 포함해라.
"""


def analyze_query(question: str, backend: Backend) -> dict:
    prompt = PROMPT_QUERY_ANALYZER.format(question=question)
    try:
        result = backend.chat_json(prompt)
        metrics.record_llm("query_analyzer")
    except LLMJsonError as e:
        logger.warning("Query Analyzer 실패, 원문 질문으로 폴백: %s", e)
        result = {}
    result.setdefault("intent", question)
    result.setdefault("keywords", [])
    result.setdefault("domain_terms", [])
    result.setdefault("required_data_type", "unknown")
    result.setdefault("candidate_filters", {})
    result.setdefault("retrieval_query", question)
    result.setdefault("reason", "자동 분석 실패로 원문 질문을 그대로 사용")
    return result


def select_documents(
    query_analysis: dict,
    question: str,
    index: HybridIndex,
    config: Config,
    *,
    min_dense_similarity: float,
    label: str = "카탈로그",
) -> list[dict]:
    """doc-first 모드(catalog/filename_only) 공용 문서 선정. `index`만 바꿔서 재사용한다."""
    query_text = query_analysis.get("retrieval_query") or question
    results = index.query(query_text, n_results=max(config.top_docs, 5))
    if not results:
        return []

    top = results[0]
    # RRF 점수는 코퍼스 내 "상대 순위"만 반영한다 — 문서 수가 적으면(카탈로그 13개)
    # 질문이 무엇이든 누군가는 항상 1등이 된다. 그래서 min_doc_score(상대 점수)만으로는
    # 완전히 무관한 질문(예: 요리 레시피)도 걸러지지 않는다. dense cosine 유사도로
    # 절대 관련성 하한선을 추가한다. 이 하한선은 모드(색인 텍스트 분포)마다 다르므로
    # 호출부가 min_dense_similarity를 명시적으로 넘긴다.
    dense_sim = top.dense_similarity or 0.0
    if dense_sim < min_dense_similarity:
        logger.info(
            "%s: 최상위 후보의 dense 유사도 부족 (%.4f < %.2f) -> 후보 문서 없음",
            label,
            dense_sim,
            min_dense_similarity,
        )
        return []

    top_score = top.score
    if top_score < config.min_doc_score:
        logger.info("%s: 최상위 점수(%.4f) < min_doc_score(%.4f) -> 후보 문서 없음", label, top_score, config.min_doc_score)
        return []

    picked = [results[0]]
    if len(results) > 1 and results[1].score >= config.doc_score_gap_ratio * top_score:
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
                "selection_reason": f"{label} 하이브리드 검색 상위 {rank}위 (dense_rank={item.dense_rank}, bm25_rank={item.bm25_rank})",
            }
        )
    return selected


def _prefilter_pages_by_text(pages: list[PageRecord], query_text: str, topn: int, tokenizer: str = "char_bigram") -> list[PageRecord]:
    from rank_bm25 import BM25Okapi

    candidates = [p for p in pages if p.char_count > 50]
    if not candidates:
        return pages[:topn]
    corpus = [tokenize_ko(p.extracted_text_if_any, tokenizer) for p in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize_ko(query_text, tokenizer))
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [p for p, _ in ranked[:topn]]


def _derive_documents_from_pages(page_results: list[ScoredItem]) -> list[dict]:
    """no_catalog: 문서 선정 단계가 없으므로 상위 페이지들의 문서를 최고 페이지 점수 순으로 역산.

    catalog_row_id는 카탈로그를 쓰지 않으므로 빈 문자열로 둔다(schema는 그대로 통과시킴).
    """
    best: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for r in page_results:
        name = r.metadata["document_name"]
        if name not in best:
            best[name] = {"score": r.score, "file_path": r.metadata["file_path"], "doc_slug": r.metadata["doc_slug"]}
            order.append(name)
        else:
            best[name]["score"] = max(best[name]["score"], r.score)

    ranked_names = sorted(order, key=lambda n: -best[n]["score"])
    return [
        {
            "rank": i,
            "document_name": name,
            "file_path": best[name]["file_path"],
            "doc_slug": best[name]["doc_slug"],
            "catalog_row_id": "",
            "selection_score": round(best[name]["score"], 6),
            "selection_reason": f"no_catalog: 전역 page_index 상위 페이지에서 역산 (최고 페이지 점수 {best[name]['score']:.4f})",
        }
        for i, name in enumerate(ranked_names, start=1)
    ]


def _ensure_doc_parsed(
    doc_slug: str,
    rel_path: str,
    config: Config,
    doc_infos: dict[str, DocumentInfo],
    page_record_lookup: dict[tuple[str, int], PageRecord],
) -> DocumentInfo:
    if doc_slug in doc_infos:
        return doc_infos[doc_slug]
    doc_info = get_or_parse_document(config.documents_dir / rel_path, rel_path, config)
    doc_infos[doc_slug] = doc_info
    for p in doc_info.pages:
        page_record_lookup[(doc_slug, p.page_number)] = p
    return doc_info


def _retrieve_pages_docfirst(
    selected_documents: list[dict],
    query_text: str,
    config: Config,
    backend: Backend,
    page_index: HybridIndex,
    *,
    top_pages: int,
    limit_pages: int | None,
) -> tuple[list[ScoredItem], list[ScoredItem], dict[tuple[str, int], PageRecord], dict[str, DocumentInfo]]:
    """catalog/filename_only 공용: 선정 문서마다 페이지를 축소 후 요약, page_index 재검색.

    반환: (top_page_results, all_page_results, page_record_lookup, doc_infos).
    all_page_results는 top_pages로 자르기 전 병합 결과 — chunk 후보 선정에 쓰인다.
    """
    doc_infos: dict[str, DocumentInfo] = {}
    page_record_lookup: dict[tuple[str, int], PageRecord] = {}
    page_results_per_doc: list[list[ScoredItem]] = []

    for doc in selected_documents:
        doc_info = _ensure_doc_parsed(doc["doc_slug"], doc["file_path"], config, doc_infos, page_record_lookup)

        if doc_info.is_scanned:
            candidate_pages = doc_info.pages[:limit_pages] if limit_pages else doc_info.pages
            logger.info(
                "%s: 스캔 문서 판정(text_ratio=%.2f) -> %d페이지 VLM 요약 대상",
                doc_info.document_name,
                doc_info.text_page_ratio,
                len(candidate_pages),
            )
        else:
            candidate_pages = _prefilter_pages_by_text(doc_info.pages, query_text, config.page_prefilter_topn, config.tokenizer)
            logger.info(
                "%s: 텍스트 문서 -> BM25 prefilter로 %d페이지 선정 (VLM 요약 대상)",
                doc_info.document_name,
                len(candidate_pages),
            )

        ensure_page_summaries(candidate_pages, doc_info.doc_slug, backend, config, page_index)
        results = page_index.query(query_text, n_results=top_pages, where={"doc_slug": doc_info.doc_slug})
        page_results_per_doc.append(results)

    all_page_results = sorted([r for lst in page_results_per_doc for r in lst], key=lambda x: -x.score)
    top_page_results = all_page_results[:top_pages]

    # ingest가 심어둔 pending_vlm placeholder가 BM25 prefilter 밖에서도 순위권에 들 수 있다
    # (원문 텍스트가 요약보다 lexical 매칭이 잘 될 때). 최종 후보에 든 것만 실제 VLM 요약으로 승격.
    for r in top_page_results:
        if not r.metadata.get("pending_vlm"):
            continue
        doc_slug_ = r.metadata["doc_slug"]
        page_num = r.metadata["page_number"]
        page_record = page_record_lookup.get((doc_slug_, page_num))
        if page_record is None:
            continue
        ensure_page_summaries([page_record], doc_slug_, backend, config, page_index)
        summary, _ = get_or_build_summary(page_record, doc_slug_, backend, config)
        r.metadata["summary"] = summary
        r.metadata["pending_vlm"] = False

    return top_page_results, all_page_results, page_record_lookup, doc_infos


def _retrieve_pages_global(
    query_text: str,
    config: Config,
    backend: Backend,
    page_index: HybridIndex,
    *,
    top_pages: int,
    prefilter_topn: int,
    min_page_dense_similarity: float,
) -> tuple[list[ScoredItem], list[ScoredItem], dict[tuple[str, int], PageRecord], dict[str, DocumentInfo]]:
    """no_catalog: page_index를 문서 범위 제한 없이(where=None) 검색.

    catalog의 2단계 깔때기(BM25 prefilter로 후보 축소 -> 후보만 VLM 요약 -> 재검색)와
    비용 예산을 맞추기 위해 여기서도: 1) 현재 표현(원문 시드 또는 기존 요약) 기준 상위
    prefilter_topn을 먼저 뽑아 2) 그 전부를 실제 VLM 요약으로 승격한 뒤 3) 승격된 표현으로
    page_index를 다시 질의해 최종 top_pages를 재선정한다. 승격은 Chroma에 upsert되므로
    두 번째 질의는 별도 처리 없이 자동으로 갱신된 표현을 반영한다.

    반환 형태는 `_retrieve_pages_docfirst`와 동일하게 맞춘다 (top, all=prefilter 후보,
    page_record_lookup, doc_infos) — run_retrieval의 chunk 단계가 모드를 몰라도 되게 하기 위함.
    """
    candidates = page_index.query(query_text, n_results=prefilter_topn, where=None)
    if not candidates:
        return [], [], {}, {}
    if (candidates[0].dense_similarity or 0.0) < min_page_dense_similarity:
        logger.info(
            "no_catalog: 최상위 페이지 dense 유사도 부족 (%.4f < %.2f) -> 후보 없음",
            candidates[0].dense_similarity or 0.0,
            min_page_dense_similarity,
        )
        return [], [], {}, {}

    doc_infos: dict[str, DocumentInfo] = {}
    page_record_lookup: dict[tuple[str, int], PageRecord] = {}
    for r in candidates:
        _ensure_doc_parsed(r.metadata["doc_slug"], r.metadata["file_path"], config, doc_infos, page_record_lookup)

    for r in candidates:  # prefilter_topn 전부를 실제 VLM 요약으로 승격 (재랭킹 전제)
        if not r.metadata.get("pending_vlm"):
            continue
        page_record = page_record_lookup.get((r.metadata["doc_slug"], r.metadata["page_number"]))
        if page_record is None:
            continue
        ensure_page_summaries([page_record], r.metadata["doc_slug"], backend, config, page_index)

    top_page_results = page_index.query(query_text, n_results=top_pages, where=None)
    # 재질의 결과가 첫 후보군 밖의 문서를 끌어올 수도 있다(승격되지 않은 페이지가 원문
    # 텍스트만으로도 여전히 상위일 때) — 그 경우를 위해 lazy하게 마저 파싱해 둔다.
    for r in top_page_results:
        _ensure_doc_parsed(r.metadata["doc_slug"], r.metadata["file_path"], config, doc_infos, page_record_lookup)

    return top_page_results, candidates, page_record_lookup, doc_infos


def _snapshot_page_index_coverage(page_index: HybridIndex) -> dict[str, Any]:
    metas = page_index.all_metadata()
    total = len(metas)
    summarized = sum(1 for m in metas if not m.get("pending_vlm", False))
    return {"total_pages": total, "summarized_pages": summarized, "pending_pages": total - summarized}


@dataclass
class RetrievalResult:
    query_analysis: dict
    selected_documents: list[dict]
    selected_pages: list[dict]
    selected_chunks: list[dict]
    document_infos: dict[str, DocumentInfo] = field(default_factory=dict)


def _build_selected_pages(top_page_results: list[ScoredItem]) -> list[dict]:
    return [
        {
            "document_name": r.metadata["document_name"],
            "page_number": r.metadata["page_number"],
            "page_score": round(r.score, 6),
            "page_summary": r.metadata.get("summary", ""),
            "image_path": r.metadata["image_path"],
        }
        for r in top_page_results
    ]


def _build_selected_chunks(top_chunk_results: list[ScoredItem], matched_terms: set[str]) -> list[dict]:
    selected_chunks = []
    for r in top_chunk_results:
        m = r.metadata
        overlap = [t for t in matched_terms if t and t in r.text]
        why = f"검색 점수 {r.score:.4f} (dense_rank={r.dense_rank}, bm25_rank={r.bm25_rank})"
        if overlap:
            why += f", 매칭 키워드: {', '.join(overlap)}"
        selected_chunks.append(
            {
                "chunk_id": r.id,
                "document_name": m["document_name"],
                "page_number": m["page_number"],
                "chunk_type": m["chunk_type"],
                "summary": m["summary"],
                "bbox": {"x1": m["bbox_x1"], "y1": m["bbox_y1"], "x2": m["bbox_x2"], "y2": m["bbox_y2"]},
                "score": round(r.score, 6),
                "why_relevant": why,
            }
        )
    return selected_chunks


def run_retrieval(
    question: str,
    config: Config,
    backend: Backend,
    *,
    mode: str | None = None,
    depth: Depth = "answer",
    top_docs: int | None = None,
    top_pages: int | None = None,
    top_chunks: int | None = None,
    limit_pages: int | None = None,
) -> RetrievalResult:
    mode = mode or config.retrieval_mode
    if mode not in (RetrievalMode.catalog, RetrievalMode.no_catalog, RetrievalMode.filename_only):
        raise ValueError(f"알 수 없는 retrieval_mode: {mode!r}")

    top_docs = top_docs or config.top_docs
    top_pages = top_pages or config.top_pages
    top_chunks = top_chunks or config.top_chunks

    page_index = get_index("page_index", config, backend)
    chunk_index = get_index("visual_chunk_index", config, backend)
    metrics.set_page_index_coverage(_snapshot_page_index_coverage(page_index))

    query_analysis = analyze_query(question, backend)
    query_text = query_analysis.get("retrieval_query") or question

    if mode in (RetrievalMode.catalog, RetrievalMode.filename_only):
        if mode == RetrievalMode.catalog:
            index = get_index("catalog_index", config, backend)
            min_sim, label = config.min_dense_similarity, "카탈로그"
        else:
            index = get_index("filename_index", config, backend)
            min_sim, label = config.min_filename_dense_similarity, "파일명"

        selected_documents = select_documents(query_analysis, question, index, config, min_dense_similarity=min_sim, label=label)[:top_docs]
        if not selected_documents:
            return RetrievalResult(query_analysis, [], [], [], {})
        if depth == "docs":
            return RetrievalResult(query_analysis, selected_documents, [], [], {})

        top_page_results, all_page_results, page_record_lookup, doc_infos = _retrieve_pages_docfirst(
            selected_documents, query_text, config, backend, page_index, top_pages=top_pages, limit_pages=limit_pages
        )
    else:  # no_catalog (page-first): 문서 선정이 없으므로 depth="docs"도 전역 페이지 질의가 필요.
        if depth == "docs":
            candidates = page_index.query(query_text, n_results=config.no_catalog_page_prefilter_topn, where=None)
            if not candidates or (candidates[0].dense_similarity or 0.0) < config.min_page_dense_similarity:
                return RetrievalResult(query_analysis, [], [], [], {})
            selected_documents = _derive_documents_from_pages(candidates)[:top_docs]
            return RetrievalResult(query_analysis, selected_documents, [], [], {})

        top_page_results, all_page_results, page_record_lookup, doc_infos = _retrieve_pages_global(
            query_text,
            config,
            backend,
            page_index,
            top_pages=top_pages,
            prefilter_topn=config.no_catalog_page_prefilter_topn,
            min_page_dense_similarity=config.min_page_dense_similarity,
        )
        if not top_page_results:
            return RetrievalResult(query_analysis, [], [], [], {})
        selected_documents = _derive_documents_from_pages(top_page_results)

    selected_pages = _build_selected_pages(top_page_results)
    if depth == "pages":
        return RetrievalResult(query_analysis, selected_documents, selected_pages, [], doc_infos)

    # chunk 단계 (3모드 공유): all_page_results(병합 전 후보 풀)에서 상위 chunk_pages_topk만
    # visual chunking 대상으로 삼는다.
    chunk_target_pages = all_page_results[: config.chunk_pages_topk]
    for r in chunk_target_pages:
        doc_slug_ = r.metadata["doc_slug"]
        page_num = r.metadata["page_number"]
        page_record = page_record_lookup.get((doc_slug_, page_num))
        if page_record is None:
            continue
        page_summary_text = r.metadata.get("summary", "")
        ensure_visual_chunks([page_record], doc_slug_, {page_num: page_summary_text}, backend, config, chunk_index)

    chunk_results_per_doc: list[list[ScoredItem]] = []
    for doc in selected_documents:
        results = chunk_index.query(query_text, n_results=top_chunks, where={"doc_slug": doc["doc_slug"]})
        chunk_results_per_doc.append(results)
    all_chunk_results = sorted([r for lst in chunk_results_per_doc for r in lst], key=lambda x: -x.score)
    top_chunk_results = all_chunk_results[:top_chunks]

    matched_terms = set(query_analysis.get("keywords", []) + query_analysis.get("domain_terms", []))
    selected_chunks = _build_selected_chunks(top_chunk_results, matched_terms)

    return RetrievalResult(
        query_analysis=query_analysis,
        selected_documents=selected_documents,
        selected_pages=selected_pages,
        selected_chunks=selected_chunks,
        document_infos=doc_infos,
    )

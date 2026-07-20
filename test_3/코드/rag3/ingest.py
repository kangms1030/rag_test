"""오프라인 1회 ingest: 카탈로그 색인 + 전 문서 파싱 -> page_index 색인.

리트리벌 경로가 아무것도 lazy로 생성하지 않도록, 여기서 표/스캔/도표 판정까지 전부
미리 끝내 Chroma에 넣어둔다(설계 문서 "오프라인 사전 인덱싱" 참고). VLM은 전혀 호출하지 않는다
(그림 캡션은 MinerU가 자체 제공하는 캡션을 쓰고, 별도 VLM 캡션은 이번 구현 범위에서 제외).
"""
from __future__ import annotations

import glob
import json
import logging
import time
from pathlib import Path
from typing import Any

from .catalog import load_catalog, match_catalog_to_pdfs, save_match_report
from .chunking import build_chunks
from .config import Config
from .flat_index import get_flat_chunk_index
from .index import get_index
from .page_store import save_page_store
from .models import Backend
from .parse import DocumentInfo, PageRecord, get_or_parse_document
from .utils import doc_slug

logger = logging.getLogger(__name__)


def _catalog_prefix_map(rows) -> dict[str, str]:
    """doc_slug -> 청크에 주입할 카탈로그 프리픽스(형제 문서 변별용, 게이트 대체).

    Phase 0/계획: 게이트는 제거하되 카탈로그의 분류/범위/키워드를 청크 텍스트에 병합해
    검색 변별력만 취한다(메타데이터 주입).
    """
    out: dict[str, str] = {}
    for row in rows:
        if not row.matched_file_path:
            continue
        slug = doc_slug(row.matched_file_path)
        name = Path(row.matched_file_path).name
        c = row.columns
        parts = [f"문서: {name}"]
        if c.get("theme"):
            parts.append(f"분류: {c['theme']}")
        if c.get("scope"):
            parts.append(f"범위: {c['scope']}")
        if c.get("keyword"):
            parts.append(f"키워드: {c['keyword']}")
        out[slug] = " | ".join(parts)
    return out


def _load_source_manifest(config: Config, slug: str) -> DocumentInfo | None:
    """청크화 소스(source_parsed_dir 우선)에서 manifest를 로드. MinerU 재파싱 회피."""
    path = config.source_parsed / slug / "manifest.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # vlm_reparse가 남긴 _ 접두 마커(_vlm_orig_text 등)는 PageRecord 필드가 아니므로 제거
    data["pages"] = [PageRecord(**{k: v for k, v in p.items() if not k.startswith("_")})
                     for p in data["pages"]]
    return DocumentInfo(**data)


def _load_content_list(config: Config, slug: str) -> tuple[list[dict], Path] | None:
    """source_parsed에서 MinerU content_list.json + images_root 반환."""
    hits = glob.glob(str(config.source_parsed / slug / "mineru" / "*" / "auto" / "*_content_list.json"))
    if not hits:
        return None
    p = Path(hits[0])
    return json.loads(p.read_text(encoding="utf-8")), p.parent


def _ingest_catalog(config: Config, backend: Backend) -> tuple[list, Any]:
    rows = load_catalog(config)
    report = match_catalog_to_pdfs(rows, config.documents_dir)
    save_match_report(report, config.output_dir / "catalog_match_report.json")

    catalog_index = get_index("catalog_index", config, backend)
    ids, texts, metas = [], [], []
    for row in rows:
        if not row.matched_file_path:
            continue
        ids.append(row.row_id)
        texts.append(row.catalog_search_text)
        metas.append(
            {
                "document_name": Path(row.matched_file_path).name,
                "file_path": row.matched_file_path,
                "doc_slug": doc_slug(row.matched_file_path),
                "title": row.columns.get("title", ""),
                "theme": row.columns.get("theme", ""),
                "publisher": row.columns.get("publisher", ""),
            }
        )
    catalog_index.upsert(ids, texts, metas)
    logger.info("catalog_index: %d개 row 색인 완료", len(ids))
    return rows, report


def _page_metadata(doc_info, page) -> dict[str, Any]:
    return {
        "document_name": page.document_name,
        "file_path": page.file_path,
        "doc_slug": page.doc_slug,
        "page_number": page.page_number,
        "page_type": page.page_type,
        "is_scanned": page.is_scanned,
        "has_table": page.has_table,
        "table_markdown": page.table_markdown,
        "table_crop_path": page.table_crop_path,
        "page_image_path": page.page_image_path,
        "figure_area_ratio": page.figure_area_ratio,
        "char_count": page.char_count,
    }


def _chunk_metadata(ch) -> dict[str, Any]:
    return {
        "document_name": ch.document_name,
        "doc_slug": ch.doc_slug,
        "page_number": ch.page_number,
        "chunk_id": ch.chunk_id,
        "block_type": ch.block_type,
        "heading_path": ch.heading_path,
        "page_type": ch.page_type,
        "is_scanned": ch.is_scanned,
        "has_table": ch.has_table,
        "figure_area_ratio": ch.figure_area_ratio,
        "table_crop_path": ch.table_crop_path,
        "page_image_path": ch.page_image_path,
        "char_count": ch.char_count,
    }


def collect_chunk_records(
    config: Config, prefix_map: dict[str, str], slug: str, doc_info: DocumentInfo,
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, int]] | None:
    """한 문서의 content_list 기반 청크 레코드(ids/indexed_texts/metas/블록타입 카운트) 생성.

    색인 텍스트 포맷(카탈로그 프리픽스 | 섹션 | p{n} + 본문)은 검색·리랭킹 품질에 직결되므로
    run_ingest(전체 재구축)와 add_doc(증분 추가)이 반드시 이 한 곳을 공유한다.
    content_list가 없으면(None) 청크 색인 불가.
    """
    cl = _load_content_list(config, slug)
    if cl is None:
        return None
    content_list, images_root = cl
    page_meta_by_num = {p.page_number: _page_metadata(doc_info, p) for p in doc_info.pages}
    chunks = build_chunks(
        content_list, doc_slug=slug, document_name=doc_info.document_name,
        page_meta=page_meta_by_num, images_root=images_root, config=config,
    )
    prefix = prefix_map.get(slug, f"문서: {doc_info.document_name}")
    ids: list[str] = []
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    for ch in chunks:
        head = f" | 섹션: {ch.heading_path}" if ch.heading_path else ""
        ids.append(ch.chunk_id)
        texts.append(f"{prefix}{head} | p{ch.page_number}\n{ch.text}")
        metas.append(_chunk_metadata(ch))
        type_counts[ch.block_type] = type_counts.get(ch.block_type, 0) + 1
    return ids, texts, metas, type_counts


def run_ingest(config: Config, backend: Backend, *, force: bool = False, limit_docs: int | None = None) -> dict[str, Any]:
    """Phase 1: page_index(big) + chunk_index(small)를 test_2 파싱 캐시에서 재사용해 구축.

    카탈로그는 게이트가 아니라 청크 프리픽스(메타데이터 주입)와 옵션 게이트용 catalog_index로만 쓴다.
    """
    config.ensure_dirs()
    t0 = time.monotonic()

    rows, report = _ingest_catalog(config, backend)
    prefix_map = _catalog_prefix_map(rows)

    matched_rows = [r for r in rows if r.matched_file_path]
    if limit_docs:
        matched_rows = matched_rows[:limit_docs]

    page_index = get_index("page_index", config, backend)
    flat_chunks = get_flat_chunk_index(config, backend)  # Chroma 비의존(B6 회피)

    total_pages = 0
    total_chunks = 0
    scanned_docs = 0
    table_pages = 0
    figure_pages = 0
    chunk_type_counts: dict[str, int] = {}
    reparsed = 0
    all_chunk_ids: list[str] = []
    all_chunk_texts: list[str] = []
    all_chunk_metas: list[dict] = []
    all_page_ids: list[str] = []
    all_page_texts: list[str] = []
    all_page_metas: list[dict] = []

    for row in matched_rows:
        rel_path = row.matched_file_path
        slug = doc_slug(rel_path)

        # 1) 소스 캐시에서 manifest 로드(없으면 MinerU 재파싱 폴백)
        doc_info = _load_source_manifest(config, slug)
        if doc_info is None or force:
            abs_path = config.documents_dir / rel_path
            doc_info = get_or_parse_document(abs_path, rel_path, config, force=force)
            reparsed += 1

        total_pages += doc_info.page_count
        if any(p.is_scanned for p in doc_info.pages):
            scanned_docs += 1
        table_pages += sum(1 for p in doc_info.pages if p.has_table)
        figure_pages += sum(1 for p in doc_info.pages if p.page_type == "figure")

        # 2) page_index (small-to-big의 big)
        page_ids = [f"{slug}_p{p.page_number:04d}" for p in doc_info.pages]
        page_texts = [p.text for p in doc_info.pages]
        page_metas = [_page_metadata(doc_info, p) for p in doc_info.pages]
        page_index.upsert(page_ids, page_texts, page_metas)
        all_page_ids.extend(page_ids)
        all_page_texts.extend(page_texts)
        all_page_metas.extend(page_metas)

        # 3) chunk_index (small) — content_list 블록 기반 청크 + 카탈로그 프리픽스 주입
        rec = collect_chunk_records(config, prefix_map, slug, doc_info)
        if rec is None:
            logger.warning("[%s] content_list 없음 -> 청크 색인 생략(page_index만)", slug)
            continue
        chunk_ids, chunk_texts, chunk_metas, type_counts = rec
        all_chunk_ids.extend(chunk_ids)
        all_chunk_texts.extend(chunk_texts)
        all_chunk_metas.extend(chunk_metas)
        for bt, n in type_counts.items():
            chunk_type_counts[bt] = chunk_type_counts.get(bt, 0) + n
        total_chunks += len(chunk_ids)

        logger.info("[%s] pages=%d chunks=%d (text=%d table=%d)", doc_info.document_name,
                    doc_info.page_count, len(chunk_ids),
                    type_counts.get("text", 0), type_counts.get("table", 0))

    # 청크 flat 인덱스 1회 빌드(전 문서 누적 -> 임베딩 -> npz+json 저장)
    flat_chunks.build(all_chunk_ids, all_chunk_texts, all_chunk_metas)
    # 페이지 텍스트 flat KV 저장(small-to-big 'big' 조회, B6 회피)
    save_page_store(config, all_page_ids, all_page_texts, all_page_metas)

    total_elapsed = time.monotonic() - t0
    summary = {
        "catalog_rows": len(rows),
        "catalog_matched": len(report.matched),
        "documents_parsed": len(matched_rows),
        "reparsed_with_mineru": reparsed,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "chunk_type_counts": chunk_type_counts,
        "avg_chunks_per_page": round(total_chunks / total_pages, 2) if total_pages else 0,
        "scanned_documents": scanned_docs,
        "table_pages": table_pages,
        "figure_pages": figure_pages,
        "page_index_count": page_index.count(),
        "chunk_index_count": flat_chunks.count(),
        "elapsed_seconds": round(total_elapsed, 2),
    }
    with open(config.output_dir / "ingest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

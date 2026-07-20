"""오프라인 1회 ingest: 카탈로그 색인 + 전 문서 파싱 -> page_index 색인.

리트리벌 경로가 아무것도 lazy로 생성하지 않도록, 여기서 표/스캔/도표 판정까지 전부
미리 끝내 Chroma에 넣어둔다(설계 문서 "오프라인 사전 인덱싱" 참고). VLM은 전혀 호출하지 않는다
(그림 캡션은 MinerU가 자체 제공하는 캡션을 쓰고, 별도 VLM 캡션은 이번 구현 범위에서 제외).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from .catalog import load_catalog, match_catalog_to_pdfs, save_match_report
from .config import Config
from .index import get_index
from .models import Backend
from .parse import get_or_parse_document
from .utils import doc_slug

logger = logging.getLogger(__name__)


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


def run_ingest(config: Config, backend: Backend, *, force: bool = False, limit_docs: int | None = None) -> dict[str, Any]:
    config.ensure_dirs()
    t0 = time.monotonic()

    rows, report = _ingest_catalog(config, backend)

    matched_rows = [r for r in rows if r.matched_file_path]
    if limit_docs:
        matched_rows = matched_rows[:limit_docs]

    page_index = get_index("page_index", config, backend)

    total_pages = 0
    scanned_docs = 0
    table_pages = 0
    figure_pages = 0
    parser_used_counts: dict[str, int] = {}

    for row in matched_rows:
        rel_path = row.matched_file_path
        abs_path = config.documents_dir / rel_path
        t_doc = time.monotonic()
        doc_info = get_or_parse_document(abs_path, rel_path, config, force=force)
        elapsed = time.monotonic() - t_doc

        total_pages += doc_info.page_count
        is_scanned_doc = any(p.is_scanned for p in doc_info.pages)
        if is_scanned_doc:
            scanned_docs += 1
        table_pages += sum(1 for p in doc_info.pages if p.has_table)
        figure_pages += sum(1 for p in doc_info.pages if p.page_type == "figure")
        parser_used_counts[doc_info.parser_used] = parser_used_counts.get(doc_info.parser_used, 0) + 1

        logger.info(
            "[%s] pages=%d parser=%s scanned=%s tables=%d figures=%d (%.1fs)",
            doc_info.document_name,
            doc_info.page_count,
            doc_info.parser_used,
            is_scanned_doc,
            sum(1 for p in doc_info.pages if p.has_table),
            sum(1 for p in doc_info.pages if p.page_type == "figure"),
            elapsed,
        )

        ids = [f"{doc_info.doc_slug}_p{p.page_number:04d}" for p in doc_info.pages]
        texts = [p.text for p in doc_info.pages]
        metas = [_page_metadata(doc_info, p) for p in doc_info.pages]
        page_index.upsert(ids, texts, metas)

    total_elapsed = time.monotonic() - t0
    summary = {
        "catalog_rows": len(rows),
        "catalog_matched": len(report.matched),
        "catalog_unmatched": len(report.unmatched_catalog_rows),
        "pdfs_unmatched": len(report.unmatched_pdfs),
        "documents_parsed": len(matched_rows),
        "total_pages": total_pages,
        "scanned_documents": scanned_docs,
        "table_pages": table_pages,
        "figure_pages": figure_pages,
        "parser_used_counts": parser_used_counts,
        "elapsed_seconds": round(total_elapsed, 2),
    }
    with open(config.output_dir / "ingest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

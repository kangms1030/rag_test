"""pdfplumber+fitz 폴백 파서. MinerU가 없거나 실패했을 때만 쓰인다.

표를 셀 구조로 정확히 뽑지 못하는 한계(설계 문서의 "표 뭉갬" 문제)가 있고 스캔 문서는
텍스트를 거의 못 뽑는다 — 그래도 파이프라인이 완전히 죽는 것보다 낫다는 최후 폴백이다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from .config import Config
from .imaging import crop_norm_bbox, save_jpeg
from .parse import DocumentInfo, PageRecord
from .utils import doc_slug

logger = logging.getLogger(__name__)

_SCANNED_TEXT_RATIO_THRESHOLD = 0.2


def _table_to_markdown(table: list[list[str | None]]) -> str:
    if not table or not table[0]:
        return ""
    rows = [[("" if c is None else str(c).replace("\n", " ")) for c in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _classify_page(fitz_page: fitz.Page, text: str, has_table: bool) -> tuple[str, float]:
    has_text = len(text.strip()) > 50
    images = fitz_page.get_images()
    drawings = fitz_page.get_drawings()
    page_area = fitz_page.rect.width * fitz_page.rect.height or 1.0

    figure_area = 0.0
    for img in images:
        try:
            rects = fitz_page.get_image_rects(img[0])
            for r in rects:
                figure_area += r.width * r.height
        except Exception:
            continue
    figure_area_ratio = min(1.0, figure_area / page_area)

    if has_table:
        return "table", figure_area_ratio
    if not has_text and (images or len(drawings) > 10):
        return "figure", figure_area_ratio
    return "text", figure_area_ratio


def parse_pdf_pdfplumber(abs_path: Path, rel_path: str, config: Config) -> DocumentInfo:
    slug = doc_slug(rel_path)
    pages_out_dir = config.parsed_dir / slug / "pages"
    crops_out_dir = config.parsed_dir / slug / "table_crops"

    fitz_doc = fitz.open(str(abs_path))
    n_pages = len(fitz_doc)
    plumber_doc = pdfplumber.open(str(abs_path))

    pages: list[PageRecord] = []
    text_page_count = 0
    try:
        for i in range(n_pages):
            page_num = i + 1
            fitz_page = fitz_doc[i]

            text = ""
            tables_md: list[str] = []
            table_bbox_norm: tuple[float, float, float, float] | None = None
            try:
                pl_page = plumber_doc.pages[i]
                text = pl_page.extract_text() or ""
                pw, ph = pl_page.width, pl_page.height
                for t in pl_page.find_tables():
                    md = _table_to_markdown(t.extract())
                    if md:
                        tables_md.append(md)
                        if table_bbox_norm is None:
                            x0, top, x1, bottom = t.bbox
                            table_bbox_norm = (x0 / pw, top / ph, x1 / pw, bottom / ph)
            except Exception as e:
                logger.warning("%s p%d pdfplumber 추출 실패: %s", rel_path, page_num, e)

            if len(text.strip()) > 50:
                text_page_count += 1

            mat = fitz.Matrix(config.answer_image_dpi / 72, config.answer_image_dpi / 72)
            pix = fitz_page.get_pixmap(matrix=mat)
            pages_out_dir.mkdir(parents=True, exist_ok=True)
            page_image_path = pages_out_dir / f"p{page_num:04d}.png"
            if not page_image_path.exists():
                pix.save(str(page_image_path))

            has_table = bool(tables_md)
            page_type, figure_area_ratio = _classify_page(fitz_page, text, has_table)

            table_crop_path = ""
            if has_table and table_bbox_norm is not None:
                crop = crop_norm_bbox(page_image_path, *table_bbox_norm)
                crop_path = crops_out_dir / f"p{page_num:04d}_table.jpg"
                save_jpeg(crop, crop_path)
                table_crop_path = str(crop_path)

            table_markdown = "\n\n".join(tables_md)
            full_text = text
            if table_markdown:
                full_text = f"{text}\n\n{table_markdown}" if text else table_markdown

            pages.append(
                PageRecord(
                    document_name=Path(rel_path).name,
                    file_path=rel_path,
                    doc_slug=slug,
                    page_number=page_num,
                    page_type=page_type,
                    text=full_text,
                    is_scanned=False,  # 문서 전체 text_ratio 계산 후 아래서 일괄 갱신
                    has_table=has_table,
                    table_markdown=table_markdown,
                    table_crop_path=table_crop_path,
                    page_image_path=str(page_image_path),
                    figure_area_ratio=figure_area_ratio,
                    char_count=len(full_text.strip()),
                )
            )
    finally:
        fitz_doc.close()
        plumber_doc.close()

    text_ratio = text_page_count / n_pages if n_pages else 0.0
    is_scanned = text_ratio < _SCANNED_TEXT_RATIO_THRESHOLD
    if is_scanned:
        for p in pages:
            p.is_scanned = True
        logger.warning(
            "%s: 스캔 문서로 판정(text_ratio=%.2f)되었으나 pdfplumber 폴백은 OCR을 지원하지 않음 "
            "-> 텍스트 검색 불가, 페이지 이미지만 근거로 남음",
            rel_path,
            text_ratio,
        )

    return DocumentInfo(
        document_name=Path(rel_path).name,
        rel_path=rel_path,
        abs_path=str(abs_path),
        doc_slug=slug,
        page_count=n_pages,
        parser_used="pdfplumber",
        pages=pages,
    )

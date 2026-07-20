"""PDF rendering + 텍스트/표 추출 + 페이지 분류.

`pdf_parse_test.ipynb`의 방식(PyMuPDF 렌더링 + pdfplumber 텍스트/표, page_type 분류)을
재사용하되, 여러 문서를 다루고 캐시를 재활용하도록 확장했다. 텍스트 추출 결과는
어디까지나 검색/축소용 보조 정보이며 최종 근거로 신뢰하지 않는다 (VLM이 근거).
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from .config import Config
from .utils import doc_slug

logger = logging.getLogger(__name__)


@dataclass
class PageRecord:
    document_name: str
    file_path: str
    page_number: int
    page_type: str  # text|visual|mixed
    extracted_text_if_any: str
    tables_if_any: list[str]
    image_path: str
    char_count: int = 0


@dataclass
class DocumentInfo:
    document_name: str
    rel_path: str
    abs_path: str
    doc_slug: str
    page_count: int
    total_pages_in_pdf: int
    text_page_ratio: float
    is_scanned: bool
    pages: list[PageRecord] = field(default_factory=list)


def classify_page(fitz_page: fitz.Page, text: str) -> str:
    has_text = len(text.strip()) > 50
    has_images = len(fitz_page.get_images()) > 0
    has_drawings = len(fitz_page.get_drawings()) > 10
    if has_text and not has_images and not has_drawings:
        return "text"
    if not has_text and (has_images or has_drawings):
        return "visual"
    if not has_text and not has_images and not has_drawings:
        return "text"
    return "mixed"


def _table_to_markdown(table: list[list[str | None]]) -> str:
    if not table or not table[0]:
        return ""
    rows = [[("" if c is None else str(c).replace("\n", " ")) for c in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_page(fitz_page: fitz.Page, page_num: int, out_dir: Path, dpi: int, *, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"p{page_num:04d}.png"
    if path.exists() and not force:
        return path
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = fitz_page.get_pixmap(matrix=mat)
    pix.save(str(path))
    return path


def parse_pdf(
    abs_path: Path,
    rel_path: str,
    config: Config,
    *,
    limit_pages: int | None = None,
    force: bool = False,
) -> DocumentInfo:
    slug = doc_slug(rel_path)
    pages_out_dir = config.pages_dir / slug

    fitz_doc = fitz.open(str(abs_path))
    total_pages = len(fitz_doc)
    n_pages = total_pages
    if limit_pages:
        n_pages = min(n_pages, limit_pages)

    plumber_doc = pdfplumber.open(str(abs_path))

    pages: list[PageRecord] = []
    text_page_count = 0
    try:
        for i in range(n_pages):
            page_num = i + 1
            fitz_page = fitz_doc[i]

            text = ""
            tables_md: list[str] = []
            try:
                pl_page = plumber_doc.pages[i]
                text = pl_page.extract_text() or ""
                for t in pl_page.extract_tables() or []:
                    md = _table_to_markdown(t)
                    if md:
                        tables_md.append(md)
            except Exception as e:
                logger.warning("%s p%d pdfplumber 추출 실패: %s", rel_path, page_num, e)

            page_type = classify_page(fitz_page, text)
            if len(text.strip()) > 50:
                text_page_count += 1

            img_path = render_page(fitz_page, page_num, pages_out_dir, config.page_render_dpi, force=force)

            pages.append(
                PageRecord(
                    document_name=Path(rel_path).name,
                    file_path=rel_path,
                    page_number=page_num,
                    page_type=page_type,
                    extracted_text_if_any=text,
                    tables_if_any=tables_md,
                    image_path=str(img_path),
                    char_count=len(text.strip()),
                )
            )
    finally:
        fitz_doc.close()
        plumber_doc.close()

    text_ratio = text_page_count / n_pages if n_pages else 0.0
    is_scanned = text_ratio < config.scanned_text_ratio_threshold

    return DocumentInfo(
        document_name=Path(rel_path).name,
        rel_path=rel_path,
        abs_path=str(abs_path),
        doc_slug=slug,
        page_count=n_pages,
        total_pages_in_pdf=total_pages,
        text_page_ratio=text_ratio,
        is_scanned=is_scanned,
        pages=pages,
    )


def _manifest_path(config: Config, slug: str) -> Path:
    return config.pages_dir / slug / "manifest.json"


def save_manifest(doc_info: DocumentInfo, config: Config) -> None:
    path = _manifest_path(config, doc_info.doc_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(doc_info), f, ensure_ascii=False, indent=2)


def load_manifest(config: Config, slug: str) -> DocumentInfo | None:
    path = _manifest_path(config, slug)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["pages"] = [PageRecord(**p) for p in data["pages"]]
    return DocumentInfo(**data)


def get_or_parse_document(
    abs_path: Path, rel_path: str, config: Config, *, limit_pages: int | None = None, force: bool = False
) -> DocumentInfo:
    """캐시된 manifest가 요청한 페이지 수를 충분히 커버하면 재사용, 아니면 재파싱.

    이미 렌더링된 page image는 `render_page`가 자체적으로 스킵하므로, 재파싱해도
    비용은 pdfplumber 텍스트/표 재추출에 국한된다.
    """
    slug = doc_slug(rel_path)
    if not force:
        cached = load_manifest(config, slug)
        if cached is not None:
            needed = min(limit_pages, cached.total_pages_in_pdf) if limit_pages else cached.total_pages_in_pdf
            if len(cached.pages) >= needed:
                return cached
    doc_info = parse_pdf(abs_path, rel_path, config, limit_pages=limit_pages, force=force)
    save_manifest(doc_info, config)
    return doc_info

"""MinerU(pipeline 백엔드) 래퍼: PDF -> content_list.json -> 페이지별 PageRecord.

pipeline 백엔드는 레이아웃+표+OCR 전용 모델(LLM/VLM 아님, 결정론적)이라 "MinerU 파서만
차용, VLM은 리트리벌에 안 씀" 결정과 정합된다. 표는 `table_body`에 HTML 셀 구조로,
스캔 페이지는 CJK OCR로 채워져 나온다(실측: core_001 DR요금표 100M/500M 숫자 100% 일치).

content_list.json 스키마(실측, mineru 3.4.4 pipeline):
  각 항목 {"type": text|table|image|chart|..., "text"|"table_body"|"img_path", "bbox":[x0,y0,x1,y1](0~1000 정규화), "page_idx"(0-based)}.
표/이미지/차트는 MinerU가 이미 크롭한 이미지를 "img_path"로 준다 — 우리가 다시 크롭할 필요 없음.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import fitz  # PyMuPDF

from .config import Config
from .parse import DocumentInfo, ParseError, PageRecord
from .utils import doc_slug

logger = logging.getLogger(__name__)

_FIGURE_TYPE_AREA_RATIO = 0.3  # 이 이상이면 page_type="figure" 후보(최종 라우팅 임계값은 config 별도)
_FIGURE_TYPE_MAX_CHARS = 200


def _resolve_device(config: Config) -> str:
    if config.mineru_device != "cuda":
        return config.mineru_device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    logger.warning("mineru_device=cuda이나 torch.cuda.is_available()=False -> cpu로 폴백")
    return "cpu"


def _run_mineru(abs_path: Path, stem: str, out_root: Path, config: Config) -> Path:
    from mineru.cli.common import do_parse, read_fn

    os.environ["MINERU_DEVICE_MODE"] = _resolve_device(config)
    pdf_bytes = read_fn(abs_path)

    do_parse(
        output_dir=str(out_root),
        pdf_file_names=[stem],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[config.mineru_lang],
        backend=config.mineru_backend,
        parse_method="auto",
        f_draw_layout_bbox=False,
        f_draw_span_bbox=False,
        f_dump_md=False,
        f_dump_middle_json=False,
        f_dump_model_output=False,
        f_dump_orig_pdf=False,
        f_dump_content_list=True,
    )

    content_list_path = out_root / stem / "auto" / f"{stem}_content_list.json"
    if not content_list_path.exists():
        raise ParseError(f"MinerU content_list.json이 생성되지 않음: {content_list_path}")
    return content_list_path


def _bbox_area_ratio(bbox: list[int] | None) -> float:
    if not bbox:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(0.0, (x1 - x0) * (y1 - y0)) / (1000.0 * 1000.0)


def _table_body_to_text(table_body_html: str) -> str:
    """표를 LLM 프롬프트에 그대로 넣을 수 있게 HTML 그대로 둔다(colspan/rowspan 보존).

    별도 마크다운 변환은 하지 않는다 — HTML도 LLM이 셀 구조를 읽기에 충분하고,
    변환 과정에서 병합 셀 정보가 유실되는 걸 피한다.
    """
    return table_body_html.strip()


def _extract_page_text(items: list[dict]) -> str:
    parts = []
    for item in items:
        t = item.get("type")
        if t == "text":
            text = item.get("text", "").strip()
            if text:
                parts.append(text)
        elif t == "table":
            captions = item.get("table_caption", [])
            footnotes = item.get("table_footnote", [])
            body = item.get("table_body", "")
            if captions:
                parts.append(" ".join(captions))
            if body:
                parts.append(_table_body_to_text(body))
            if footnotes:
                parts.append(" ".join(footnotes))
        elif t in ("chart",):
            content = item.get("content", "")
            if content:
                parts.append(str(content))
    return "\n\n".join(parts)


def _pick_representative_table(items: list[dict]) -> dict | None:
    tables = [it for it in items if it.get("type") == "table" and it.get("table_body")]
    if not tables:
        return None
    return max(tables, key=lambda it: _bbox_area_ratio(it.get("bbox")))


def _render_page_image(fitz_page: fitz.Page, page_num: int, out_dir: Path, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"p{page_num:04d}.png"
    if not path.exists():
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = fitz_page.get_pixmap(matrix=mat)
        pix.save(str(path))
    return path


def parse_pdf_mineru(abs_path: Path, rel_path: str, config: Config) -> DocumentInfo:
    slug = doc_slug(rel_path)
    out_root = config.parsed_dir / slug / "mineru"
    stem = slug

    try:
        content_list_path = _run_mineru(abs_path, stem, out_root, config)
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"MinerU 파싱 실패({rel_path}): {e}") from e

    with open(content_list_path, "r", encoding="utf-8") as f:
        content_list: list[dict] = json.load(f)

    images_root = content_list_path.parent  # {out_root}/{stem}/auto — img_path는 이 기준 상대경로

    by_page: dict[int, list[dict]] = {}
    for item in content_list:
        by_page.setdefault(item.get("page_idx", 0), []).append(item)

    fitz_doc = fitz.open(str(abs_path))
    n_pages = len(fitz_doc)
    pages_out_dir = config.parsed_dir / slug / "pages"

    pages: list[PageRecord] = []
    try:
        for i in range(n_pages):
            page_num = i + 1
            items = by_page.get(i, [])
            fitz_page = fitz_doc[i]

            native_text = (fitz_page.get_text() or "").strip()
            is_scanned = len(native_text) < 50

            text = _extract_page_text(items)
            table = _pick_representative_table(items)
            has_table = table is not None
            table_markdown = _table_body_to_text(table["table_body"]) if table else ""
            table_crop_path = ""
            if table and table.get("img_path"):
                candidate = images_root / table["img_path"]
                if candidate.exists():
                    table_crop_path = str(candidate)

            figure_area_ratio = sum(
                _bbox_area_ratio(it.get("bbox")) for it in items if it.get("type") in ("image", "chart")
            )
            figure_area_ratio = min(1.0, figure_area_ratio)

            char_count = len(text.strip())
            if has_table:
                page_type = "table"
            elif figure_area_ratio >= _FIGURE_TYPE_AREA_RATIO and char_count < _FIGURE_TYPE_MAX_CHARS:
                page_type = "figure"
            else:
                page_type = "text"

            page_image_path = _render_page_image(fitz_page, page_num, pages_out_dir, config.answer_image_dpi)

            pages.append(
                PageRecord(
                    document_name=Path(rel_path).name,
                    file_path=rel_path,
                    doc_slug=slug,
                    page_number=page_num,
                    page_type=page_type,
                    text=text,
                    is_scanned=is_scanned,
                    has_table=has_table,
                    table_markdown=table_markdown,
                    table_crop_path=table_crop_path,
                    page_image_path=str(page_image_path),
                    figure_area_ratio=figure_area_ratio,
                    char_count=char_count,
                )
            )
    finally:
        fitz_doc.close()

    return DocumentInfo(
        document_name=Path(rel_path).name,
        rel_path=rel_path,
        abs_path=str(abs_path),
        doc_slug=slug,
        page_count=n_pages,
        parser_used="mineru",
        pages=pages,
    )

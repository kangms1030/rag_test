"""PDF -> 페이지별 구조화 텍스트 레코드. 파서(MinerU/Docling/pdfplumber)는 인터페이스
뒤에 있어 교체 가능하다 — `Parser.parse()`가 같은 `DocumentInfo`를 반환하기만 하면
ingest.py/retrieve.py/answer.py는 어떤 파서가 쓰였는지 몰라도 된다.

MinerU가 기본(config.parser="mineru")이고, 표를 HTML/마크다운 셀 구조로, 스캔 페이지는
CJK OCR로 텍스트화해 pdfplumber의 표 뭉갬·스캔 무력화 문제를 해소한다. GPU(CUDA) 파싱
실패 시 CPU로, MinerU 자체가 없으면 pdfplumber 폴백으로 내려간다(나머지 파이프라인 무변경).
"""
from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .utils import doc_slug

logger = logging.getLogger(__name__)


@dataclass
class PageRecord:
    document_name: str
    file_path: str
    doc_slug: str
    page_number: int
    #: text|table|figure — 그림 위주(면적 대부분+텍스트 적음)면 "figure", 표가 있으면 "table",
    #: 그 외 "text". 라우팅(retrieve.py)이 이 값을 결정론적으로 본다.
    page_type: str
    text: str  # 페이지 text 블록 + 표 마크다운(+스캔이면 OCR 텍스트) 전부 합친 검색/독해용 본문
    is_scanned: bool
    has_table: bool
    table_markdown: str
    #: 표가 있으면 그 표만 크롭한 고해상도 이미지 경로(비전 경로용). 없으면 "".
    table_crop_path: str
    #: 페이지 전체 렌더 이미지 경로(비전 폴백/근거 표시용).
    page_image_path: str
    #: 순수 그림 페이지 판정용: figure 블록 면적 합 / 페이지 면적.
    figure_area_ratio: float
    char_count: int = 0


@dataclass
class DocumentInfo:
    document_name: str
    rel_path: str
    abs_path: str
    doc_slug: str
    page_count: int
    parser_used: str
    pages: list[PageRecord] = field(default_factory=list)


class ParseError(RuntimeError):
    pass


def _manifest_path(config: Config, slug: str) -> Path:
    return config.parsed_dir / slug / "manifest.json"


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


def _parse_with_mineru(abs_path: Path, rel_path: str, config: Config) -> DocumentInfo:
    from .parse_mineru import parse_pdf_mineru

    return parse_pdf_mineru(abs_path, rel_path, config)


def _parse_with_pdfplumber(abs_path: Path, rel_path: str, config: Config) -> DocumentInfo:
    from .parse_pdfplumber import parse_pdf_pdfplumber

    return parse_pdf_pdfplumber(abs_path, rel_path, config)


def parse_document(abs_path: Path, rel_path: str, config: Config) -> DocumentInfo:
    """config.parser 우선 시도, 실패 시 pdfplumber로 폴백(폴백해도 나머지 파이프라인 무변경)."""
    if config.parser == "mineru":
        try:
            return _parse_with_mineru(abs_path, rel_path, config)
        except Exception as e:
            logger.error("%s: MinerU 파싱 실패(%s) -> pdfplumber 폴백", rel_path, e)
            return _parse_with_pdfplumber(abs_path, rel_path, config)
    return _parse_with_pdfplumber(abs_path, rel_path, config)


def get_or_parse_document(abs_path: Path, rel_path: str, config: Config, *, force: bool = False) -> DocumentInfo:
    """캐시된 manifest가 있으면 재사용(재파싱 없이 산출물 재사용), 아니면 새로 파싱해 캐시에 저장.

    ingest 때 전부 선연산해두는 설계이므로 ask 경로에서는 이 함수가 항상 캐시 히트여야 한다.
    """
    slug = doc_slug(rel_path)
    if not force:
        cached = load_manifest(config, slug)
        if cached is not None:
            return cached
    doc_info = parse_document(abs_path, rel_path, config)
    save_manifest(doc_info, config)
    return doc_info

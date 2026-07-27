"""Excel 데이터 카탈로그 ingestion.

pandas로 시트를 읽는다. 헤더에 리터럴 `\n`이 포함되어 있으므로(`"dct:title\n(파일명)"`)
컬럼명을 그대로 보존해야 한다. 셀 값의 NaN은 빈 값으로 취급한다.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from .config import Config

logger = logging.getLogger(__name__)

# 정규화된(공백 정리) 헤더 -> 역할. 여러 키워드가 매치되면 먼저 나온 규칙이 우선.
_COLUMN_ROLE_KEYWORDS: dict[str, list[str]] = {
    "title": ["dct:title", "파일명", "제목"],
    "theme": ["dcat:theme", "대분류", "분류"],
    "publisher": ["dct:publisher", "발행", "출처", "기관"],
    "issued": ["dct:issued", "작성연도", "버전"],
    "purpose": ["dct:purpose", "활용목적", "용도"],
    "description": ["dct:description", "설명"],
    "keyword": ["dcat:keyword", "키워드"],
    "scope": ["문서 범위", "document_scope", "범위"],
    "format": ["dct:format", "형식"],
    "folder": ["연결 dataset", "폴더명", "데이터셋명"],
    "sample_questions": ["대표 예상 질문", "예상 질문"],
    "download_url": ["downloadurl", "드라이브 링크"],
    "latest": ["최신여부"],
}


def _normalize_header(h: Any) -> str:
    if h is None:
        return ""
    return re.sub(r"\s+", " ", str(h)).strip()


_EXT_RE = re.compile(r"\.(pdf|hwpx|hwp|xlsx|xls|docx|doc|txt)$", re.IGNORECASE)


def _normalize_filename(name: str) -> str:
    """Korean NFC/NFD 및 특수문자 차이를 흡수하기 위한 파일명 정규화.

    확장자는 구두점을 지우기 '전에' 떼어낸다 (카탈로그에 원본 포맷(.hwpx 등)이
    기록되고 배포본은 .pdf인 경우가 있어, 먼저 떼지 않으면 구두점 제거 후
    확장자 문자열이 본문에 들러붙어 두 파일명이 서로 달라져 버린다).
    """
    n = unicodedata.normalize("NFC", name)
    n = _EXT_RE.sub("", n)
    n = n.casefold()
    n = re.sub(r"[★_\-\.\(\)\[\]【】·, ]+", "", n)
    return n


@dataclass
class CatalogRow:
    row_id: str
    sheet: str
    raw: dict[str, Any]
    columns: dict[str, str] = field(default_factory=dict)  # role -> value
    catalog_search_text: str = ""
    matched_file_path: str | None = None
    match_method: str | None = None  # exact|basename|fuzzy
    match_score: float = 0.0


def _infer_column_roles(headers: list[str]) -> dict[str, str]:
    """정규화된 헤더 리스트에서 role -> 원본 헤더 매핑을 추론."""
    normalized = [_normalize_header(h) for h in headers]
    lowered = [h.lower() for h in normalized]
    roles: dict[str, str] = {}
    for role, keywords in _COLUMN_ROLE_KEYWORDS.items():
        for h_orig, h_norm, h_low in zip(headers, normalized, lowered):
            if h_orig is None or h_norm in roles.values():
                continue
            if any(kw in h_low for kw in keywords):
                roles[role] = h_orig
                break
    return roles


def _read_sheet_rows(path: Path, sheet_name: str) -> tuple[list[str], list[dict[str, Any]]]:
    """pandas로 한 시트를 읽어 (원본 헤더 목록, row dict 목록)을 반환.

    헤더의 `\n`은 보존하고, NaN 셀은 None으로 정규화한다. 빈 행은 제외한다.
    """
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    except ValueError as e:  # 시트명 오류를 명확한 메시지로 변환
        available = pd.ExcelFile(path).sheet_names
        raise ValueError(f"시트 '{sheet_name}'를 읽을 수 없음({e}). 존재하는 시트: {available}") from e

    headers = [str(c) for c in df.columns]
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = {str(col): (None if pd.isna(val) else val) for col, val in row.items()}
        if all(v is None for v in rec.values()):
            continue
        records.append(rec)
    return headers, records


def load_catalog(config: Config) -> list[CatalogRow]:
    logger.info("Excel 카탈로그 로드: %s [시트: %s]", config.catalog_excel_path, config.catalog_sheet)
    headers, records = _read_sheet_rows(config.catalog_excel_path, config.catalog_sheet)
    roles = _infer_column_roles(headers)
    logger.info("컬럼 역할 추론 결과: %s", json.dumps(roles, ensure_ascii=False))

    missing = [r for r in ("title", "description", "keyword") if r not in roles]
    if missing:
        logger.warning("추론하지 못한 중요 컬럼 역할: %s (config.yaml catalog_columns로 오버라이드 가능)", missing)

    rows: list[CatalogRow] = []
    for idx, r in enumerate(records):
        col_values = {role: str(r[col]).strip() for role, col in roles.items() if r.get(col) not in (None, "")}
        title = col_values.get("title", "")
        if not title:
            continue

        text_parts = []
        if title:
            text_parts.append(f"제목: {title}")
        if col_values.get("theme"):
            text_parts.append(f"분류: {col_values['theme']}")
        if col_values.get("scope"):
            text_parts.append(f"범위: {col_values['scope']}")
        if col_values.get("description"):
            text_parts.append(f"설명: {col_values['description']}")
        if col_values.get("keyword"):
            text_parts.append(f"키워드: {col_values['keyword']}")
        if col_values.get("sample_questions"):
            text_parts.append(f"대표질문: {col_values['sample_questions']}")

        search_text = " | ".join(text_parts)

        row_id = f"cat_{idx:04d}"
        rows.append(
            CatalogRow(
                row_id=row_id,
                sheet=config.catalog_sheet,
                raw={str(k): v for k, v in r.items()},
                columns=col_values,
                catalog_search_text=search_text,
            )
        )
    logger.info("카탈로그 row %d개 로드 완료", len(rows))
    return rows


@dataclass
class MatchReport:
    matched: list[dict[str, Any]]
    unmatched_catalog_rows: list[dict[str, Any]]
    unmatched_pdfs: list[str]


def match_catalog_to_pdfs(
    rows: list[CatalogRow], documents_dir: Path, fuzzy_threshold: float = 88.0
) -> MatchReport:
    pdf_paths = sorted(documents_dir.rglob("*.pdf"))
    norm_to_path: dict[str, Path] = {}
    for p in pdf_paths:
        norm = _normalize_filename(p.name)
        norm_to_path[norm] = p

    used_pdfs: set[str] = set()
    matched: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for row in rows:
        title = row.columns.get("title", "")
        norm_title = _normalize_filename(title)

        found: Path | None = None
        method: str | None = None
        score = 0.0

        # 1) exact normalized match
        if norm_title in norm_to_path:
            found = norm_to_path[norm_title]
            method, score = "exact", 100.0
        else:
            # 2) basename contains / is-contained-by
            for norm_name, p in norm_to_path.items():
                if norm_title and (norm_title in norm_name or norm_name in norm_title):
                    found = p
                    method, score = "basename", 95.0
                    break
        if found is None:
            # 3) fuzzy match
            best_score = 0.0
            best_path: Path | None = None
            for norm_name, p in norm_to_path.items():
                s = fuzz.token_set_ratio(norm_title, norm_name)
                if s > best_score:
                    best_score, best_path = s, p
            if best_path is not None and best_score >= fuzzy_threshold:
                found, method, score = best_path, "fuzzy", best_score

        if found is not None:
            rel = str(found.relative_to(documents_dir))
            row.matched_file_path = rel
            row.match_method = method
            row.match_score = score
            used_pdfs.add(rel)
            matched.append(
                {
                    "row_id": row.row_id,
                    "title": title,
                    "file_path": rel,
                    "method": method,
                    "score": score,
                }
            )
        else:
            unmatched_rows.append({"row_id": row.row_id, "title": title})

    unmatched_pdfs = [str(p.relative_to(documents_dir)) for p in pdf_paths if str(p.relative_to(documents_dir)) not in used_pdfs]

    logger.info(
        "카탈로그-PDF 매칭: %d/%d row 매칭, PDF %d개 중 %d개 미사용",
        len(matched),
        len(rows),
        len(pdf_paths),
        len(unmatched_pdfs),
    )
    return MatchReport(matched=matched, unmatched_catalog_rows=unmatched_rows, unmatched_pdfs=unmatched_pdfs)


def save_match_report(report: MatchReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "matched": report.matched,
                "unmatched_catalog_rows": report.unmatched_catalog_rows,
                "unmatched_pdfs": report.unmatched_pdfs,
                "summary": {
                    "matched_count": len(report.matched),
                    "unmatched_catalog_rows_count": len(report.unmatched_catalog_rows),
                    "unmatched_pdfs_count": len(report.unmatched_pdfs),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("매칭 리포트 저장: %s", output_path)

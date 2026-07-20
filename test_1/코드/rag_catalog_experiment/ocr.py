"""evidence_text를 채우기 위한 텍스트 추출. 오직 사후 근거 보강에만 쓴다.

의도적으로 page_index 구축에는 절대 연결하지 않는다 — OCR로 스캔 페이지를 텍스트
검색 가능하게 만들면 그건 카탈로그의 스캔 문서 비용 우위를 없애는 또 다른 baseline이
되어 "카탈로그가 스캔 문서에서 비용을 줄이는가"라는 실험 질문 자체가 바뀐다.

우선순위: pdfplumber 좌표 교차(정확, 텍스트 레이어 있는 문서에만 동작) ->
(config.enable_ocr가 켜져 있을 때만) pytesseract OCR. pytesseract 미설치 시 경고 1회만
찍고 파이프라인은 계속 진행한다(하드 실패시키지 않는다).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_tesseract_warned = False


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401

        return True
    except ImportError:
        return False


def extract_text_from_pdfplumber(abs_pdf_path: Path, page_number: int, bbox: dict[str, float]) -> str:
    """bbox(페이지 비율 0~1) 안에 위치 좌표가 들어오는 단어들을 pdfplumber로 모아 반환."""
    import pdfplumber

    try:
        with pdfplumber.open(str(abs_pdf_path)) as pdf:
            if not (1 <= page_number <= len(pdf.pages)):
                return ""
            page = pdf.pages[page_number - 1]
            x1, y1 = bbox["x1"] * page.width, bbox["y1"] * page.height
            x2, y2 = bbox["x2"] * page.width, bbox["y2"] * page.height
            words = page.extract_words()
    except Exception as e:
        logger.warning("pdfplumber 좌표 텍스트 추출 실패(%s p%d): %s", abs_pdf_path, page_number, e)
        return ""

    hits = [w["text"] for w in words if x1 <= w["x0"] <= x2 and y1 <= w["top"] <= y2]
    return " ".join(hits)


def extract_text_ocr(image_path: Path, bbox: dict[str, float]) -> str:
    global _tesseract_warned
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        if not _tesseract_warned:
            logger.warning("enable_ocr=true이나 pytesseract 미설치 -> OCR 건너뜀 (evidence_text는 빈 채로 유지)")
            _tesseract_warned = True
        return ""

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            box = (int(bbox["x1"] * w), int(bbox["y1"] * h), int(bbox["x2"] * w), int(bbox["y2"] * h))
            crop = img.crop(box)
            return pytesseract.image_to_string(crop, lang="kor+eng").strip()
    except Exception as e:
        logger.warning("OCR 실패(%s): %s", image_path, e)
        return ""


def fill_evidence_text(evidence: dict[str, Any], *, config: Any, abs_pdf_path: Path | None) -> str:
    """evidence_text가 이미 있으면 그대로 두고, 없으면 pdfplumber -> (선택) OCR 순서로 채운다."""
    if evidence.get("evidence_text", "").strip():
        return evidence["evidence_text"]

    bbox = evidence.get("bbox", {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})

    if abs_pdf_path is not None:
        text = extract_text_from_pdfplumber(abs_pdf_path, evidence["page_number"], bbox)
        if text.strip():
            return text

    if getattr(config, "enable_ocr", False):
        text = extract_text_ocr(Path(evidence["crop_image_path"]), {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})
        if text:
            return text

    return ""

"""답변 생성 포크 — P1(c) 적응형 컨텍스트 트림 + (P3) 합성/문장인용은 여기에 얹는다.

원본: rag3/answer.py — PROMPT/포맷/전사추출/폴백 로직은 전부 import 재사용.
변경 요지: x_adaptive_trim=True면 rerank 점수가 top 대비 급락하는 페이지를 답변 컨텍스트에서
제외해 prefill을 줄인다(1순위 페이지는 항상 보존). 플래그 OFF면 rag3.answer와 완전 동일.
"""
from __future__ import annotations

import logging
from typing import Any

from rag3.answer import answer_text_from_pages as _answer_text_from_pages
from rag3.answer import answer_vision_from_page  # noqa: F401  (변경 없음 — 재수출)

logger = logging.getLogger(__name__)


def _trim_pages_by_score(pages: list[dict[str, Any]], config: Any) -> list[dict[str, Any]]:
    """rerank(page_score) top 대비 drop_ratio 미만 페이지를 제외. 최소 1페이지는 보존."""
    if not pages:
        return pages
    ratio = float(getattr(config, "x_adaptive_trim_drop_ratio", 0.5))
    top = float(pages[0].get("page_score", 0.0) or 0.0)
    if top <= 0:
        return pages
    kept = [pages[0]]
    for p in pages[1:]:
        if float(p.get("page_score", 0.0) or 0.0) >= ratio * top:
            kept.append(p)
    return kept


def answer_text_from_pages_x(question: str, pages: list[dict[str, Any]], backend, config) -> dict[str, Any]:
    """P1(c) 적응형 트림 후 원본 답변 로직 호출. 플래그 OFF면 원본과 동일."""
    if getattr(config, "x_adaptive_trim", False):
        pages = _trim_pages_by_score(pages, config)
    return _answer_text_from_pages(question, pages, backend, config)

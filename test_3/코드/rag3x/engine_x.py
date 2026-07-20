"""Rag3xEngine — rag3.Rag3Engine의 실험판 미러. controller_x + 백엔드 선택만 다르다.

원본: rag3/engine.py (Rag3Engine). ask() 반환은 기존 키 유지 + 추가만(API 계약 불변).
전 실험 플래그 OFF + x_backend=ollama면 Rag3Engine과 동일 동작.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag3.config import Config

from .backends import get_x_backend
from .xconfig import load_x_config

logger = logging.getLogger(__name__)


class Rag3xEngine:
    def __init__(self, config_path: str | Path | None = None,
                 overrides: dict[str, Any] | None = None,
                 x_overrides: dict[str, Any] | None = None, *, preload: bool = True):
        self.config: Config = load_x_config(config_path, overrides, x_overrides)
        self.config.ensure_dirs()
        self.backend = get_x_backend(self.config)
        if preload:
            self.warm_up()

    def warm_up(self, *, deep: bool = False) -> None:
        from rag3.rerank import get_reranker
        from rag3.flat_index import get_flat_chunk_index
        from rag3.page_store import load_page_store
        get_reranker(self.config)
        n_chunks = get_flat_chunk_index(self.config, self.backend).count()
        n_pages = len(load_page_store(self.config))
        logger.info("[rag3x] 인덱스 로드: 청크 %d · 페이지 %d · backend=%s", n_chunks, n_pages,
                    getattr(self.config, "x_backend", "ollama"))
        if deep:
            self.backend.embed(["워밍업"], is_query=True)
            self.backend.chat_text("답변은 '준비'라고만 하세요.")

    def ask(self, question: str, *, run_id: str | None = None,
            resolve_images: bool = True, save_evidence: bool = False) -> dict[str, Any]:
        from .controller_x import answer_question
        from rag3.utils import new_run_id
        rid = run_id or new_run_id(question)
        result = answer_question(question, self.config, self.backend, run_id=rid)
        if resolve_images or save_evidence:
            from rag3.evidence import export_evidence, resolve_evidence
            resolve_evidence(result, self.config)
            if save_evidence:
                result["evidence_files"] = export_evidence(result, self.config, rid)
        return result

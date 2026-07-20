"""RAG3를 단일 객체로 사용하는 모듈 진입점 (LangGraph 등 외부 오케스트레이터 연동용).

사용 예:
    from rag3 import Rag3Engine
    engine = Rag3Engine()              # 프로세스당 1회 생성 (모델 웜 로딩)
    result = engine.ask("질문...")      # AskResult dict — 스키마는 아래 TypedDict 참고

ask()의 반환은 controller.answer_question의 dict를 그대로 유지하고
(기존 소비자와 호환), evidence 이미지 절대경로 등 추가 키만 부착한다.
torch/sentence_transformers 등 무거운 의존성은 메서드 안에서 lazy import해
`import rag3` 자체는 가볍게 유지한다(python -m rag3 check 등).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from .config import Config, load_config

logger = logging.getLogger(__name__)


class EvidenceItem(TypedDict, total=False):
    document_name: str
    page_number: int
    page_image_path: str          # 색인 당시 저장된 경로(stale 가능)
    table_crop_path: str
    page_image_resolved: str | None   # 존재 확인된 절대경로 (resolve_images=True)
    table_crop_resolved: str | None


class AskResult(TypedDict, total=False):
    run_id: str
    question: str
    answer_path: str              # "text" | "vision" | "none"
    final_answer: str
    confidence: str               # "high" | "low" | "abstain" | "unknown"
    selected_documents: list[dict[str, Any]]
    selected_pages: list[dict[str, Any]]   # evidence와 같은 순서(page_score 포함)
    evidence: list[EvidenceItem]
    rerank_top_score: float | None
    route_reason: str
    verification: dict[str, Any] | None
    rollback_history: list[dict[str, Any]]
    metrics: dict[str, Any]       # 모델 호출수 + timings_seconds{retrieve,answer,total}
    evidence_files: list[dict[str, Any]]   # save_evidence=True일 때만


class Rag3Engine:
    """설정/백엔드/인덱스를 1회 로드하고 질문마다 완성 파이프라인을 실행한다."""

    def __init__(self, config_path: str | Path | None = None,
                 overrides: dict[str, Any] | None = None, *, preload: bool = True):
        self.config: Config = load_config(config_path, overrides)
        self.config.ensure_dirs()
        from .models import get_backend
        self.backend = get_backend(self.config)
        if preload:
            self.warm_up()

    def warm_up(self, *, deep: bool = False) -> None:
        """리랭커/인덱스를 선로드해 첫 질문 지연과 환경 오류를 앞당긴다.

        deep=True면 임베딩·LLM에 1회 더미 호출을 보내 Ollama 모델을 VRAM에 상주시킨다
        (keep_alive 30m — 발표 직전 재호출 권장).
        """
        from .rerank import get_reranker
        logger.info("리랭커 로딩...")
        get_reranker(self.config)
        from .flat_index import get_flat_chunk_index
        from .page_store import load_page_store
        n_chunks = get_flat_chunk_index(self.config, self.backend).count()  # kiwi 토큰화+BM25 구축 포함
        n_pages = len(load_page_store(self.config))
        logger.info("인덱스 로드 완료: 청크 %d · 페이지 %d", n_chunks, n_pages)
        if deep:
            logger.info("딥 워밍업: 임베딩 + LLM 1토큰 호출...")
            self.backend.embed(["워밍업"], is_query=True)
            self.backend.chat_text("답변은 '준비'라고만 하세요.")
            logger.info("딥 워밍업 완료")

    def ask(self, question: str, *, run_id: str | None = None,
            resolve_images: bool = True, save_evidence: bool = False) -> AskResult:
        """질문 1건 처리: 검색 → 답변 → 검증 → 롤백 (controller.answer_question)."""
        from .controller import answer_question
        from .utils import new_run_id
        rid = run_id or new_run_id(question)
        result = answer_question(question, self.config, self.backend, run_id=rid)
        if resolve_images or save_evidence:
            from .evidence import export_evidence, resolve_evidence
            resolve_evidence(result, self.config)
            if save_evidence:
                result["evidence_files"] = export_evidence(result, self.config, rid)
        return result  # type: ignore[return-value]

    def health(self) -> dict[str, Any]:
        """코퍼스/모델 현황 — 웹 데모 및 연동 측 헬스체크용."""
        from .flat_index import get_flat_chunk_index
        from .page_store import load_page_store
        store = load_page_store(self.config)
        doc_slugs = {rec.get("meta", {}).get("doc_slug", "") for rec in store.values()}
        doc_slugs.discard("")
        return {
            "chunks": get_flat_chunk_index(self.config, self.backend).count(),
            "pages": len(store),
            "documents": len(doc_slugs),
            "models": {
                "text": self.config.text_answer_model,
                "vision": self.config.vision_answer_model,
                "embedding": self.config.embedding_model,
                "reranker": self.config.rerank_model,
            },
            "index_dir": str(self.config.index_dir),
        }

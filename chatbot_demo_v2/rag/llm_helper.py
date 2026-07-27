"""경량 LLM 헬퍼 — contextualize/composer/grader 등 '작은' LLM 노드 전용.

RAG 엔진(리랭커·인덱스 로드)과 분리된 **채팅 전용** 백엔드를 지연 생성한다. 따라서 FAQ/시나리오
경로는 이 헬퍼를 호출하지 않는 한 GPU/인덱스 없이 동작한다(결정론 경로 LLM 0회 원칙 유지).

- 백엔드: rag3x.get_x_backend(config) — RAG 백엔드와 동일 종류(gemini flash-lite). GeminiBackend 는
  프로세스 전역 rate-limit(_last_call_ts)을 클래스변수로 공유하므로 RAG 백엔드와 호출 간격을 함께 지킨다.
- 실패(키 없음/네트워크)해도 예외를 밖으로 던지지 않는다 → 노드가 pass-through 로 안전 저하.
- 프롬프트 조립은 노드가 담당(prompts/ 파일 + PromptLoader). 이 클래스는 chat() 프리미티브만 제공.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from ..config.settings import Settings
from .adapter_util import prepare_ragcore_imports

logger = logging.getLogger("chatbot_demo_v2.llm")


class LlmHelper:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._backend = None
        self._ready = False
        self._failed = False
        self._lock = threading.Lock()
        self.calls = 0          # 관측용(테스트/health)

    @property
    def available(self) -> bool:
        return not self._failed

    def _ensure(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._ready:
                return True
            if self._failed:
                return False
            try:
                prepare_ragcore_imports(self._settings)
                from rag3x.xconfig import load_x_config
                from rag3x.backends import get_x_backend

                config = load_x_config(
                    str(self._settings.ragcore_config),
                    x_overrides={"x_backend": self._settings.rag_backend},
                )
                self._backend = get_x_backend(config)
                self._ready = True
                return True
            except Exception:  # noqa: BLE001
                logger.warning(
                    "LlmHelper 백엔드 준비 실패 — 소형 LLM 노드는 pass-through 로 동작합니다."
                )
                self._failed = True
                return False

    def chat(self, prompt: str) -> Optional[str]:
        """단발 텍스트 생성. 실패 시 None(노드가 pass-through 판단)."""
        if not self._ensure():
            return None
        try:
            self.calls += 1
            out = self._backend.chat_text(prompt)
            return (out or "").strip() or None
        except Exception:  # noqa: BLE001
            return None

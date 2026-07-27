"""백엔드 팩토리 — x_backend 설정으로 로컬(ollama) / Gemini Flash를 선택.

- "ollama": rag3.models.OllamaBackend 그대로(등가성).
- "gemini": gemini_backend.GeminiBackend (생성·검증만 Flash, 임베딩은 로컬 위임 — P2).
"""
from __future__ import annotations

from rag3.config import Config
from rag3.models import Backend, get_backend as _get_ollama_backend


def get_x_backend(config: Config) -> Backend:
    kind = getattr(config, "x_backend", "ollama")
    if kind == "ollama":
        return _get_ollama_backend(config)
    if kind == "gemini":
        from .gemini_backend import GeminiBackend
        return GeminiBackend(config)
    raise ValueError(f"지원하지 않는 x_backend: {kind}")

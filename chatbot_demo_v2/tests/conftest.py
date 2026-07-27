"""테스트 공용 픽스처. 외부 API/GPU/LangSmith 없이 실행 가능."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot_demo_v2.config.settings import load_settings
from chatbot_demo_v2.app.dependencies import build_context
from chatbot_demo_v2.rag.adapter_util import FakeRagAdapter


def _base_env(tmp_path: Path | None = None) -> dict:
    env = {
        "RAG_BACKEND": "gemini",
        "SCENARIO_MATCH_THRESHOLD": "0.90",
        "SCENARIO_MATCH_MARGIN": "0.05",
        "CLARIFY_ENABLED": "true",
        "CLARIFY_MIN_SCORE": "0.75",
        "COMPOSER_RAG_ENABLED": "true",
        "COMPOSER_FAQ_ENABLED": "true",
        "CONTEXTUALIZE_ENABLED": "true",
        "GRADER_ENABLED": "true",
        "WEB_SEARCH_ENABLED": "false",
        "WEB_SEARCH_SCOPE": "in_domain_unresolved",
        "LANGSMITH_TRACING": "false",
        "DEMO_PORT": "8002",
    }
    if tmp_path is not None:
        env["DEMO_EVIDENCE_DIR"] = str(tmp_path / "evidence")
    return env


@pytest.fixture
def settings(tmp_path):
    return load_settings(env=_base_env(tmp_path))


@pytest.fixture
def fake_rag():
    return FakeRagAdapter()


class OfflineLlm:
    """기본 테스트용 LLM — 항상 미응답(None).

    compose/grader/contextualize 가 pass-through 로 안전 저하되는지까지 함께 검증한다.
    실제 네트워크 호출이 없어 테스트가 빠르고 결정론적이다.
    """

    def __init__(self):
        self.calls = 0

    def chat(self, prompt: str):
        self.calls += 1
        return None


@pytest.fixture
def offline_llm():
    return OfflineLlm()


@pytest.fixture
def ctx(settings, fake_rag, offline_llm):
    return build_context(settings, rag_adapter=fake_rag, llm=offline_llm)


@pytest.fixture
def client(ctx):
    from fastapi.testclient import TestClient
    from chatbot_demo_v2.app.main import create_app

    app = create_app(ctx)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def faq(ctx):
    return ctx.faq


@pytest.fixture
def tree(ctx):
    return ctx.tree

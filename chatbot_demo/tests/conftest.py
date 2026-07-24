"""테스트 공용 픽스처. 외부 API/GPU/LangSmith 없이 실행 가능."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot_demo.config.settings import load_settings
from chatbot_demo.app.dependencies import build_context
from chatbot_demo.rag.rag3x_adapter import FakeRagAdapter


# 실제 데이터 파일을 사용하는 설정(키·엔진 불필요)
def _base_env(tmp_path: Path | None = None) -> dict:
    env = {
        "RAG3X_BACKEND": "gemini",
        "SCENARIO_MATCH_THRESHOLD": "0.90",
        "SCENARIO_MATCH_MARGIN": "0.05",
        "WEB_SEARCH_ENABLED": "false",
        "WEB_SEARCH_SCOPE": "in_domain_unresolved",
        "LANGSMITH_TRACING": "false",
        "DEMO_PORT": "8001",
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


@pytest.fixture
def ctx(settings, fake_rag):
    return build_context(settings, rag_adapter=fake_rag)


@pytest.fixture
def client(ctx):
    from fastapi.testclient import TestClient
    from chatbot_demo.app.main import create_app

    app = create_app(ctx)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def faq(ctx):
    return ctx.faq


@pytest.fixture
def tree(ctx):
    return ctx.tree

"""FastAPI 엔드포인트: health/root/chat/reset/warmup, 오류코드, 키 미노출."""

from __future__ import annotations

from chatbot_demo.config.settings import load_settings
from chatbot_demo.app.dependencies import build_context
from chatbot_demo.app.main import create_app
from chatbot_demo.rag.rag3x_adapter import FakeRagAdapter


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "engine" in body and "langsmith" in body and "web_search" in body


def test_scenarios_root(client):
    r = client.get("/api/scenarios/root")
    assert r.status_code == 200
    assert r.json()["node_id"] == "root"
    assert len(r.json()["options"]) == 9


def test_chat_button_flow(client):
    r = client.post("/api/chat", json={
        "action": {"type": "scenario_option", "node_id": "root",
                   "option_id": "internet_down", "label": "인터넷이 안 돼요"}})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "scenario"
    assert body["session_id"]
    assert len(body["options"]) == 4


def test_chat_empty_message_400(client):
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400


def test_chat_neither_message_nor_action_400(client):
    assert client.post("/api/chat", json={}).status_code == 400


def test_chat_invalid_action_400(client):
    r = client.post("/api/chat", json={
        "action": {"type": "scenario_option", "node_id": "root",
                   "option_id": "NOPE", "label": "x"}})
    assert r.status_code == 400


def test_reset_and_warmup(client):
    r = client.post("/api/chat", json={"message": "스쿨넷이 뭐예요?"})
    sid = r.json()["session_id"]
    assert client.post("/api/reset", json={"session_id": sid}).status_code == 200
    assert client.post("/api/warmup", json={}).status_code == 200


def test_concurrent_rag_returns_429(settings):
    busy_rag = FakeRagAdapter(raise_busy=True)
    app = create_app(build_context(settings, rag_adapter=busy_rag))
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"message": "답 못하는 질문 xyz 12345"})
        assert r.status_code == 429


def test_rag_unavailable_returns_503(settings):
    unavail = FakeRagAdapter(raise_unavailable=True)
    app = create_app(build_context(settings, rag_adapter=unavail))
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/chat", json={"message": "답 못하는 질문 abc 98765"})
        assert r.status_code == 503


def test_app_starts_without_langsmith_key(settings):
    """LangSmith 키 없이도 앱 생성/health 200."""
    app = create_app(build_context(settings, rag_adapter=FakeRagAdapter()))
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        assert c.get("/api/health").json()["langsmith"]["tracing_enabled"] is False


def test_api_key_not_leaked_in_responses(monkeypatch, settings):
    """가짜 GEMINI/LANGSMITH 키가 응답 어디에도 노출되지 않아야 함."""
    secret = "SUPERSECRETKEY_abcdef1234567890"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setenv("LANGSMITH_API_KEY", secret)
    app = create_app(build_context(settings, rag_adapter=FakeRagAdapter()))
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        blobs = []
        blobs.append(c.get("/api/health").text)
        blobs.append(c.get("/api/scenarios/root").text)
        blobs.append(c.post("/api/chat", json={
            "action": {"type": "scenario_option", "node_id": "root",
                       "option_id": "callcenter", "label": "콜센터"}}).text)
        blobs.append(c.post("/api/chat", json={"message": "스쿨넷이 뭐예요?"}).text)
        for b in blobs:
            assert secret not in b


def test_evidence_path_traversal_blocked(client):
    assert client.get("/evidence/..%2f..%2fetc/passwd").status_code in (404, 400)
    assert client.get("/evidence/goodrun/does_not_exist.png").status_code == 404

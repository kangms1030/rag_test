"""SSE 스트리밍(/api/chat/stream) + 피드백(/api/feedback)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from chatbot_demo_v2.app.dependencies import build_context
from chatbot_demo_v2.app.main import create_app
from chatbot_demo_v2.config.settings import load_settings
from chatbot_demo_v2.rag.adapter_util import FakeRagAdapter
from chatbot_demo_v2.scenario.models import MatchResult


class _OfflineLlm:
    def chat(self, prompt: str):
        return None


def _client(settings, **kw):
    kw.setdefault("llm", _OfflineLlm())
    kw.setdefault("rag_adapter", FakeRagAdapter())
    ctx = build_context(settings, **kw)
    return TestClient(create_app(ctx)), ctx


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """'event: X\\ndata: {...}\\n\\n' 블록들을 [(event, data)] 로."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                ev = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if ev:
            out.append((ev, data))
    return out


def _settings(tmp_path, **over):
    env = {"DEMO_EVIDENCE_DIR": str(tmp_path / "ev"), "CONTEXTUALIZE_ENABLED": "false"}
    env.update(over)
    return load_settings(env=env)


def test_stream_faq_emits_progress_nodes_and_final(tmp_path):
    c, ctx = _client(_settings(tmp_path))
    q = ctx.faq.entries[0].question
    with c.stream("POST", "/api/chat/stream", json={"message": q}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse("".join(r.iter_text()))

    kinds = [e for e, _ in events]
    assert kinds[0] == "progress"          # 시작 진행상황
    assert "node" in kinds                 # 노드 진행 이벤트
    assert kinds[-1] == "final"            # 마지막은 최종 응답

    final = dict(events[-1][1])
    assert final["answer"] == ctx.faq.entries[0].answer
    assert final["answer_source"] == "faq_match"
    assert final["run_id"]

    nodes = [d["node"] for e, d in events if e == "node"]
    assert "normalize_input" in nodes and "final_formatter" in nodes


def test_stream_rag_emits_rag_progress(tmp_path):
    """RAG 경로는 어댑터에 progress 콜백이 전달돼 내부 단계도 스트리밍된다."""
    c, _ = _client(_settings(tmp_path))
    with c.stream("POST", "/api/chat/stream",
                  json={"message": "코퍼스에 없는 아주 생소한 질문 zzz 98765"}) as r:
        events = _parse_sse("".join(r.iter_text()))
    stages = [d.get("stage") for e, d in events if e == "progress"]
    assert "rag" in stages          # 부모 노드 발신
    assert "retrieve" in stages     # 어댑터 콜백을 통한 내부 단계(fake)
    assert events[-1][0] == "final"


def test_stream_clarify_pauses(tmp_path):
    c, ctx = _client(_settings(tmp_path))
    e0, e1 = ctx.faq.entries[0], ctx.faq.entries[1]

    def fake_match(norm):
        return MatchResult(
            decision="reject_ambiguous", decision_reason="t", best_score=0.85,
            second_score=0.84, margin_observed=0.01, threshold=0.90, margin_required=0.05,
            matched_id=e0.id, matched_question=e0.question,
            matched_sheet=e0.sheet, matched_row=e0.row,
        )

    ctx.matcher.match = fake_match
    ctx.matcher.top_candidates = lambda norm, k=2: [
        {"faq_id": e0.id, "question": e0.question, "score": 0.85},
        {"faq_id": e1.id, "question": e1.question, "score": 0.84},
    ]

    with c.stream("POST", "/api/chat/stream", json={"message": "애매한 질문"}) as r:
        events = _parse_sse("".join(r.iter_text()))
    assert events[-1][0] == "clarify"                    # final 대신 clarify 로 종료
    payload = events[-1][1]
    assert {c_["faq_id"] for c_ in payload["candidates"]} == {e0.id, e1.id}
    assert payload["session_id"]


def test_stream_resume_after_clarify(tmp_path):
    c, ctx = _client(_settings(tmp_path))
    e0 = ctx.faq.entries[0]
    ctx.matcher.match = lambda norm: MatchResult(
        decision="reject_ambiguous", decision_reason="t", best_score=0.85,
        second_score=0.84, margin_observed=0.01, threshold=0.90, margin_required=0.05,
        matched_id=e0.id, matched_question=e0.question,
        matched_sheet=e0.sheet, matched_row=e0.row,
    )
    ctx.matcher.top_candidates = lambda norm, k=2: [
        {"faq_id": e0.id, "question": e0.question, "score": 0.85}
    ]

    with c.stream("POST", "/api/chat/stream", json={"message": "애매한 질문"}) as r:
        events = _parse_sse("".join(r.iter_text()))
    sid = events[-1][1]["session_id"]

    with c.stream("POST", "/api/chat/stream",
                  json={"session_id": sid, "clarify_response": {"choice": e0.id}}) as r2:
        events2 = _parse_sse("".join(r2.iter_text()))
    assert events2[-1][0] == "final"
    assert events2[-1][1]["answer"] == e0.answer


def test_stream_rag_busy_emits_error_event(tmp_path):
    c, _ = _client(_settings(tmp_path), rag_adapter=FakeRagAdapter(raise_busy=True))
    with c.stream("POST", "/api/chat/stream", json={"message": "답 못하는 질문 xyz 12345"}) as r:
        events = _parse_sse("".join(r.iter_text()))
    assert events[-1][0] == "error"
    assert events[-1][1]["status"] == 429


def test_stream_invalid_request_400(tmp_path):
    c, _ = _client(_settings(tmp_path))
    r = c.post("/api/chat/stream", json={})          # message/action 둘 다 없음
    assert r.status_code == 400


def test_feedback_noop_without_langsmith(tmp_path):
    c, _ = _client(_settings(tmp_path))
    r = c.post("/api/feedback", json={"run_id": "00000000-0000-0000-0000-000000000000",
                                      "score": 1, "comment": "좋아요"})
    assert r.status_code == 200
    assert r.json()["recorded"] is False        # 추적 비활성 → no-op

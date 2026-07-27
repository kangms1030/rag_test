"""대화 메모리 + contextualize: 같은 세션에서 이력 누적, 후속질문 재작성 게이트.

composer/grader 는 꺼서 contextualize 동작만 격리 검증한다(합성은 test_composer 담당).
"""

from __future__ import annotations

from chatbot_demo_v2.app.dependencies import build_context
from chatbot_demo_v2.config.settings import load_settings
from chatbot_demo_v2.rag.adapter_util import FakeRagAdapter


class FakeLlm:
    """chat(prompt) 프리미티브만 제공. 호출 횟수 기록."""

    def __init__(self, reply="재작성된 독립 질문"):
        self.calls = 0
        self._reply = reply

    def chat(self, prompt: str):
        self.calls += 1
        return self._reply


def _settings(tmp_path, **over):
    env = {
        "COMPOSER_FAQ_ENABLED": "false",
        "COMPOSER_RAG_ENABLED": "false",
        "GRADER_ENABLED": "false",
        "DEMO_EVIDENCE_DIR": str(tmp_path / "ev"),
    }
    env.update(over)
    return load_settings(env=env)


def _text(msg, sid, tid):
    return {"session_id": sid, "thread_id": tid, "input_type": "text", "user_input": msg}


def _cfg(tid):
    return {"configurable": {"thread_id": tid}}


def test_first_turn_no_contextualize(tmp_path):
    llm = FakeLlm()
    ctx = build_context(_settings(tmp_path), rag_adapter=FakeRagAdapter(), llm=llm)
    q = ctx.faq.entries[0].question  # 정확일치 FAQ
    r = ctx.graph.invoke(_text(q, "m1", "tm1"), _cfg("tm1"))
    assert llm.calls == 0                       # 첫 턴은 재작성 안 함
    assert r["contextualized"] is False
    assert len(r["messages"]) == 2              # 사용자 + 상담봇


def test_followup_triggers_contextualize(tmp_path):
    llm = FakeLlm(reply="스쿨넷 회선 장애 조치 방법")
    ctx = build_context(_settings(tmp_path), rag_adapter=FakeRagAdapter(), llm=llm)
    q1 = ctx.faq.entries[0].question
    ctx.graph.invoke(_text(q1, "m2", "tm2"), _cfg("tm2"))                      # 턴1(FAQ)
    r2 = ctx.graph.invoke(_text("그건 어떻게 해결해?", "m2", "tm2"), _cfg("tm2"))  # 턴2(후속)
    assert llm.calls == 1                       # 후속질문에서만 재작성 호출
    assert r2["contextualized"] is True
    assert r2["standalone_question"] == "스쿨넷 회선 장애 조치 방법"
    assert len(r2["messages"]) == 4             # 턴1(2) + 턴2(2)


def test_contextualize_disabled(tmp_path):
    llm = FakeLlm()
    s = _settings(tmp_path, CONTEXTUALIZE_ENABLED="false")
    ctx = build_context(s, rag_adapter=FakeRagAdapter(), llm=llm)
    q1 = ctx.faq.entries[0].question
    ctx.graph.invoke(_text(q1, "m3", "tm3"), _cfg("tm3"))
    r2 = ctx.graph.invoke(_text("그건 어떻게?", "m3", "tm3"), _cfg("tm3"))
    assert llm.calls == 0                       # 토글 OFF → 재작성 안 함
    assert r2["contextualized"] is False

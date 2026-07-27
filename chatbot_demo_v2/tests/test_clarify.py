"""clarify(HITL 인터럽트): 애매 매칭 시 되묻고, 재개(Command)로 FAQ/RAG 분기."""

from __future__ import annotations

import pytest
from langgraph.types import Command

from chatbot_demo_v2.app.dependencies import build_context
from chatbot_demo_v2.rag.adapter_util import FakeRagAdapter
from chatbot_demo_v2.scenario.models import MatchResult


class FakeLlm:
    """오프라인 LLM(항상 미응답) — clarify 분기만 격리 검증(합성은 test_composer 담당)."""

    def chat(self, prompt: str):
        return None


def _force_ambiguous(ctx, best_score=0.85):
    """ctx.matcher 를 애매(reject_ambiguous, 고점) + 상위 후보 2개로 강제."""
    e0, e1 = ctx.faq.entries[0], ctx.faq.entries[1]

    def fake_match(norm):
        return MatchResult(
            decision="reject_ambiguous", decision_reason="테스트 강제",
            best_score=best_score, second_score=best_score - 0.01,
            margin_observed=0.01, threshold=0.90, margin_required=0.05,
            matched_id=e0.id, matched_question=e0.question,
            matched_sheet=e0.sheet, matched_row=e0.row,
        )

    def fake_top(norm, k=2):
        return [
            {"faq_id": e0.id, "question": e0.question, "score": best_score},
            {"faq_id": e1.id, "question": e1.question, "score": best_score - 0.01},
        ]

    ctx.matcher.match = fake_match
    ctx.matcher.top_candidates = fake_top
    return e0, e1


def _text(msg, sid, tid):
    return {"session_id": sid, "thread_id": tid, "input_type": "text", "user_input": msg}


def _cfg(tid):
    return {"configurable": {"thread_id": tid}}


def test_ambiguous_triggers_interrupt(settings):
    ctx = build_context(settings, rag_adapter=FakeRagAdapter(), llm=FakeLlm())
    e0, e1 = _force_ambiguous(ctx)
    out = ctx.graph.invoke(_text("인터넷 관련 애매한 질문", "c1", "tc1"), _cfg("tc1"))
    assert "__interrupt__" in out                         # 일시정지됨
    payload = out["__interrupt__"][0].value
    assert payload["type"] == "clarify"
    ids = {c["faq_id"] for c in payload["candidates"]}
    assert e0.id in ids and e1.id in ids


def test_resume_pick_candidate_goes_faq(settings):
    ctx = build_context(settings, rag_adapter=FakeRagAdapter(), llm=FakeLlm())
    e0, _ = _force_ambiguous(ctx)
    ctx.graph.invoke(_text("애매한 질문", "c2", "tc2"), _cfg("tc2"))        # → interrupt
    r = ctx.graph.invoke(Command(resume={"choice": e0.id}), _cfg("tc2"))    # 후보 선택
    assert r["route"] == "faq"
    assert r["answer_source"] == "faq_match"
    assert r["final_answer"] == e0.answer


def test_resume_none_goes_rag(settings):
    fake_rag = FakeRagAdapter()
    ctx = build_context(settings, rag_adapter=fake_rag, llm=FakeLlm())
    _force_ambiguous(ctx)
    ctx.graph.invoke(_text("애매한 질문", "c3", "tc3"), _cfg("tc3"))         # → interrupt
    r = ctx.graph.invoke(Command(resume={"choice": "__none__"}), _cfg("tc3"))
    assert r["route"] == "rag3x"
    assert fake_rag.ask_calls == 1
    assert r["answer_source"] == "rag3x"


def test_clarify_disabled_no_interrupt(tmp_path):
    from chatbot_demo_v2.config.settings import load_settings
    s = load_settings(env={"CLARIFY_ENABLED": "false",
                           "DEMO_EVIDENCE_DIR": str(tmp_path / "ev")})
    fake_rag = FakeRagAdapter()
    ctx = build_context(s, rag_adapter=fake_rag, llm=FakeLlm())
    _force_ambiguous(ctx)
    out = ctx.graph.invoke(_text("애매한 질문", "c4", "tc4"), _cfg("tc4"))
    assert "__interrupt__" not in out                     # 되묻지 않고
    assert out["route"] == "rag3x"                         # 바로 RAG

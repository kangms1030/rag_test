"""LangGraph 통합: 버튼 네비게이션, 세션 격리, FAQ 무LLM, abstain/web 라우팅."""

from __future__ import annotations

from chatbot_demo.config.settings import load_settings
from chatbot_demo.app.dependencies import build_context
from chatbot_demo.rag.rag3x_adapter import FakeRagAdapter
from chatbot_demo.web_search.mock import MockWebSearchProvider


def _run(ctx, state, thread_id):
    cfg = {"configurable": {"thread_id": thread_id}}
    return ctx.graph.invoke(state, cfg)


def _action(node_id, option_id, label, sid="s", tid="s:0"):
    return {
        "session_id": sid, "thread_id": tid, "input_type": "action",
        "action_type": "scenario_option", "action_node_id": node_id,
        "selected_option_id": option_id, "action_label": label,
    }


def _text(msg, sid="s", tid="s:0"):
    return {"session_id": sid, "thread_id": tid, "input_type": "text", "user_input": msg}


def test_button_navigation_terminal(ctx):
    r = _run(ctx, _action("root", "internet_down", "인터넷이 안 돼요"), "t1")
    assert r["route"] == "scenario"
    assert r["current_node_id"] == "internet_down.situation"
    r = _run(ctx, _action("internet_down.situation", "school_all", "학교 전체", "s", "t1"), "t1")
    r = _run(ctx, _action("internet_down.all.duration", "over_15", "15분 이상", "s", "t1"), "t1")
    assert r["scenario_completed"] is True
    assert r["final_answer"].startswith("15분 이상")
    assert r["answer_source"] == "scenario_tree"


def test_scenario_answer_calls_no_llm(ctx, fake_rag):
    """시나리오/FAQ 답변 경로에서 RAG(=LLM 대체) 어댑터가 호출되지 않아야 함."""
    _run(ctx, _action("root", "tablet_power", "태블릿 전원"), "t2")
    faq_q = ctx.faq.entries[0].question
    _run(ctx, _text(faq_q, "s2", "t3"), "t3")
    assert fake_rag.ask_calls == 0


def test_free_text_faq_exact_match(ctx):
    q = ctx.faq.entries[0].question
    r = _run(ctx, _text(q, "s3", "t4"), "t4")
    assert r["route"] == "scenario"
    assert r["answer_source"] == "faq_match"
    assert r["final_answer"] == ctx.faq.entries[0].answer  # 원문 그대로
    assert r["scenario_match"]["decision"] == "exact"


def test_free_text_low_similarity_goes_rag(ctx, fake_rag):
    r = _run(ctx, _text("오늘 저녁 뭐 먹을지 골라줘", "s4", "t5"), "t5")
    assert r["route"] == "rag3x"
    assert fake_rag.ask_calls == 1
    assert r["answer_source"] == "rag3x"


def test_session_isolation(ctx):
    # 서로 다른 thread_id 는 시나리오 상태가 분리됨
    _run(ctx, _action("root", "internet_down", "인터넷이 안 돼요", "A", "tA"), "tA")
    rB = _run(ctx, _action("root", "internet_slow", "인터넷이 느려요", "B", "tB"), "tB")
    assert rB["current_node_id"] == "internet_slow.scope"
    # A 스레드는 여전히 자기 위치에서 진행 가능
    rA = _run(ctx, _action("internet_down.situation", "wifi", "와이파이", "A", "tA"), "tA")
    assert rA["current_node_id"] == "wifi.ap_light"


def test_rag_abstain_web_disabled_yields_abstain(settings):
    abstain_rag = FakeRagAdapter(result={
        "run_id": "r", "final_answer": "", "answer_path": "none",
        "confidence": "abstain", "verification": {"abstain": True},
        "evidence": [], "metrics": {}, "selected_pages": [],
    })
    ctx = build_context(settings, rag_adapter=abstain_rag)
    r = _run(ctx, _text("우리 학교 회선 계약 언제 만료돼?", "s5", "t6"), "t6")
    assert r["route"] == "abstain"
    assert r["confidence"] == "abstain"
    assert r["warnings"]


def test_rag_abstain_web_enabled_calls_mock(tmp_path):
    env = {
        "WEB_SEARCH_ENABLED": "true", "WEB_SEARCH_SCOPE": "in_domain_unresolved",
        "DEMO_EVIDENCE_DIR": str(tmp_path / "ev"),
    }
    settings = load_settings(env=env)
    abstain_rag = FakeRagAdapter(result={
        "run_id": "r", "final_answer": "", "answer_path": "none",
        "confidence": "unknown", "verification": None,
        "evidence": [], "metrics": {}, "selected_pages": [],
    })
    mock_web = MockWebSearchProvider(canned_answer="웹 답변")
    ctx = build_context(settings, rag_adapter=abstain_rag, web_provider=mock_web)
    r = _run(ctx, _text("아무거나 답 못하는 질문 12345", "s6", "t7"), "t7")
    assert r["route"] == "web_search"
    assert mock_web.call_count == 1
    assert r["answer_source"] == "web"
    assert r["final_answer"] == "웹 답변"


def test_disabled_web_provider_never_called(ctx):
    """기본(Disabled) provider 는 abstain 경로에서도 호출되지 않는다."""
    # ctx 는 web_search 비활성 → route 는 web_search 로 가지 않음
    called = {"n": 0}
    orig = ctx.web_provider.search_and_answer

    def spy(*a, **k):
        called["n"] += 1
        return orig(*a, **k)

    ctx.web_provider.search_and_answer = spy
    _run(ctx, _text("답 못하는 질문 zzz 999", "s7", "t8"), "t8")
    assert called["n"] == 0

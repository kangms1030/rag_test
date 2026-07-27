"""compose_answer(근거 종합) + answer_grader(해결도 판정·에스컬레이션 사이클).

원칙 검증: 시나리오 버튼은 합성하지 않고, 합성이 근거 밖 수치를 만들면 폐기(원문 복귀),
FAQ 미해결이면 RAG로 1회만 에스컬레이션.
"""

from __future__ import annotations

from chatbot_demo_v2.app.dependencies import build_context
from chatbot_demo_v2.config.settings import load_settings
from chatbot_demo_v2.rag.adapter_util import FakeRagAdapter


class ScriptedLlm:
    """프롬프트 내용으로 composer/grader 를 구분해 정해진 응답을 준다."""

    def __init__(self, compose=None, grade=None):
        self._compose = compose
        self._grade = grade
        self.compose_calls = 0
        self.grade_calls = 0

    def chat(self, prompt: str):
        # answer_grader.md 에만 'UNRESOLVED' 토큰이 들어 있다.
        if "UNRESOLVED" in prompt:
            self.grade_calls += 1
            return self._grade
        self.compose_calls += 1
        return self._compose


def _settings(tmp_path, **over):
    env = {"DEMO_EVIDENCE_DIR": str(tmp_path / "ev"),
           "CONTEXTUALIZE_ENABLED": "false"}
    env.update(over)
    return load_settings(env=env)


def _text(msg, sid="s", tid="t"):
    return {"session_id": sid, "thread_id": tid, "input_type": "text", "user_input": msg}


def _cfg(tid="t"):
    return {"configurable": {"thread_id": tid}}


def _action(node_id, option_id, label, tid="t"):
    return {"session_id": "s", "thread_id": tid, "input_type": "action",
            "action_type": "scenario_option", "action_node_id": node_id,
            "selected_option_id": option_id, "action_label": label}


# ---------- compose (FAQ) ----------

def test_faq_answer_is_composed(tmp_path):
    llm = ScriptedLlm(compose="정리된 상담체 답변입니다.")
    s = _settings(tmp_path, GRADER_ENABLED="false")
    ctx = build_context(s, rag_adapter=FakeRagAdapter(), llm=llm)
    q = ctx.faq.entries[0].question
    r = ctx.graph.invoke(_text(q), _cfg())
    assert r["composed"] is True
    assert r["final_answer"] == "정리된 상담체 답변입니다."
    assert r["original_answer"] == ctx.faq.entries[0].answer   # 원문 동봉
    assert llm.compose_calls == 1


def test_composer_disabled_returns_original(tmp_path):
    llm = ScriptedLlm(compose="합성되면 안 됨")
    s = _settings(tmp_path, COMPOSER_FAQ_ENABLED="false", GRADER_ENABLED="false")
    ctx = build_context(s, rag_adapter=FakeRagAdapter(), llm=llm)
    e0 = ctx.faq.entries[0]
    r = ctx.graph.invoke(_text(e0.question), _cfg())
    assert r["composed"] is False
    assert r["final_answer"] == e0.answer
    assert llm.compose_calls == 0


def test_composer_fallback_on_unsupported_number(tmp_path):
    """근거에 없는 수치를 지어내면 합성을 폐기하고 원문으로 복귀한다(LLM 0회 결정론 대조)."""
    llm = ScriptedLlm(compose="AP는 총 987654대 설치되어 있습니다.")
    s = _settings(tmp_path, GRADER_ENABLED="false")
    ctx = build_context(s, rag_adapter=FakeRagAdapter(), llm=llm)
    e0 = ctx.faq.entries[0]
    r = ctx.graph.invoke(_text(e0.question), _cfg())
    assert r["composed"] is False
    assert r["final_answer"] == e0.answer                  # 원문 유지
    assert "987654" in (r["composer_fallback"] or "")      # 폐기 사유 기록


def test_composer_allows_numbers_from_prompt_boilerplate(tmp_path):
    """프롬프트가 지시한 안내 상수(지원센터 1899-0979)를 인용해도 폐기하면 안 된다.

    실측 회귀: 이 허용이 없으면 composer_rag 프롬프트의 '스쿨넷 지원센터(1899-0979)' 지시를
    모델이 따랐을 때 1899/0979 가 '근거 밖 수치'로 오탐돼 정상 합성이 전부 폐기됐다.
    """
    llm = ScriptedLlm(
        compose="정확한 조치는 스쿨넷 서비스 지원센터(1899-0979)로 문의해 주세요."
    )
    s = _settings(tmp_path, GRADER_ENABLED="false")
    rag = FakeRagAdapter(result={
        "run_id": "r", "final_answer": "초안", "answer_context": "근거 텍스트",
        "answer_path": "text", "confidence": "high", "verification": {"abstain": False},
        "evidence": [], "metrics": {}, "selected_pages": [],
    })
    ctx = build_context(s, rag_adapter=rag, llm=llm)
    r = ctx.graph.invoke(_text("생소한 질문 zzz 45678"), _cfg())
    assert r["composed"] is True, r.get("composer_fallback")
    assert "1899-0979" in r["final_answer"]


def test_scenario_button_answer_is_not_composed(tmp_path):
    """시나리오 버튼 종단답변은 절차 안내라 원문 그대로(합성 금지)."""
    llm = ScriptedLlm(compose="합성되면 안 됨")
    s = _settings(tmp_path)
    ctx = build_context(s, rag_adapter=FakeRagAdapter(), llm=llm)
    ctx.graph.invoke(_action("root", "internet_down", "인터넷이 안 돼요"), _cfg("tb"))
    ctx.graph.invoke(_action("internet_down.situation", "school_all", "학교 전체", "tb"), _cfg("tb"))
    r = ctx.graph.invoke(_action("internet_down.all.duration", "over_15", "15분 이상", "tb"), _cfg("tb"))
    assert r["final_answer"].startswith("15분 이상")
    assert r["composed"] is False
    assert llm.compose_calls == 0


# ---------- compose (RAG) ----------

def test_rag_answer_is_composed(tmp_path):
    llm = ScriptedLlm(compose="근거를 종합한 최종 답변입니다.")
    s = _settings(tmp_path, GRADER_ENABLED="false")
    rag = FakeRagAdapter(result={
        "run_id": "r", "final_answer": "초안 답변", "answer_context": "근거 페이지 텍스트",
        "answer_path": "text", "confidence": "high", "verification": {"abstain": False},
        "evidence": [], "metrics": {}, "selected_pages": [],
    })
    ctx = build_context(s, rag_adapter=rag, llm=llm)
    r = ctx.graph.invoke(_text("코퍼스에 없는 아주 생소한 질문 zzz 12345"), _cfg())
    assert r["route"] == "rag3x"
    assert r["composed"] is True
    assert r["final_answer"] == "근거를 종합한 최종 답변입니다."


def test_rag_abstain_is_not_composed(tmp_path):
    """답변 자체가 없으면(abstain) 합성 노드를 타지 않는다."""
    llm = ScriptedLlm(compose="합성되면 안 됨")
    s = _settings(tmp_path)
    rag = FakeRagAdapter(result={
        "run_id": "r", "final_answer": "", "answer_path": "none",
        "confidence": "abstain", "verification": {"abstain": True},
        "evidence": [], "metrics": {}, "selected_pages": [],
    })
    ctx = build_context(s, rag_adapter=rag, llm=llm)
    r = ctx.graph.invoke(_text("답 못하는 질문 zzz 999"), _cfg())
    assert r["route"] == "abstain"
    assert llm.compose_calls == 0


# ---------- grader + 에스컬레이션 사이클 ----------

def test_grader_escalates_faq_to_rag_once(tmp_path):
    """FAQ 답변이 미해결이면 RAG로 1회 재시도(메인그래프 사이클)."""
    llm = ScriptedLlm(compose=None, grade="UNRESOLVED")
    s = _settings(tmp_path, COMPOSER_FAQ_ENABLED="false", COMPOSER_RAG_ENABLED="false")
    rag = FakeRagAdapter()
    ctx = build_context(s, rag_adapter=rag, llm=llm)
    q = ctx.faq.entries[0].question
    r = ctx.graph.invoke(_text(q), _cfg())
    assert r["grader_verdict"] == "unresolved"
    assert rag.ask_calls == 1                 # RAG 재시도 1회
    assert r["answer_source"] == "rag3x"      # 최종은 RAG 답변
    assert r["escalate_budget"] == 0          # 예산 소진(무한루프 방지)
    assert llm.grade_calls == 1               # RAG 경로에선 grader 생략


def test_grader_resolved_keeps_faq_answer(tmp_path):
    llm = ScriptedLlm(compose=None, grade="RESOLVED")
    s = _settings(tmp_path, COMPOSER_FAQ_ENABLED="false")
    rag = FakeRagAdapter()
    ctx = build_context(s, rag_adapter=rag, llm=llm)
    e0 = ctx.faq.entries[0]
    r = ctx.graph.invoke(_text(e0.question), _cfg())
    assert r["grader_verdict"] == "resolved"
    assert rag.ask_calls == 0                 # 에스컬레이션 없음
    assert r["final_answer"] == e0.answer


def test_grader_disabled_no_escalation(tmp_path):
    llm = ScriptedLlm(compose=None, grade="UNRESOLVED")
    s = _settings(tmp_path, COMPOSER_FAQ_ENABLED="false", GRADER_ENABLED="false")
    rag = FakeRagAdapter()
    ctx = build_context(s, rag_adapter=rag, llm=llm)
    e0 = ctx.faq.entries[0]
    r = ctx.graph.invoke(_text(e0.question), _cfg())
    assert rag.ask_calls == 0
    assert r["final_answer"] == e0.answer
    assert llm.grade_calls == 0

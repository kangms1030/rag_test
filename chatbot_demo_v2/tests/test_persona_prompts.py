"""독립 페르소나 프롬프트의 렌더링·토글·LangGraph 연결 회귀 테스트."""

from __future__ import annotations

from chatbot_demo_v2.app.dependencies import build_context
from chatbot_demo_v2.config.settings import PKG_ROOT, load_settings
from chatbot_demo_v2.prompts.loader import PromptLoader
from chatbot_demo_v2.rag.adapter_util import FakeRagAdapter


class CaptureLlm:
    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, prompt: str):
        self.prompts.append(prompt)
        return "근거를 바탕으로 정리한 답변입니다."


def _rag_result():
    return {
        "run_id": "r",
        "final_answer": "초안 답변",
        "answer_context": "근거 페이지 텍스트",
        "answer_path": "text",
        "confidence": "high",
        "verification": {"abstain": False},
        "evidence": [],
        "metrics": {},
        "selected_pages": [],
    }


def test_persona_prompt_files_render_and_missing_is_safe(tmp_path):
    prompts = PromptLoader(PKG_ROOT / "prompts")
    assert "학교 유·무선 네트워크" in prompts.render_optional("persona/persona")
    assert "현재 답변 경로는 rag3x" in prompts.render_optional(
        "persona/response_policy", route="rag3x"
    )

    missing = PromptLoader(tmp_path)
    assert missing.render_optional("persona/not-installed") == ""


def test_persona_prompts_can_be_disabled():
    settings = load_settings(env={"PERSONA_PROMPTS_ENABLED": "false"})
    assert settings.persona_prompts_enabled is False


def test_persona_prompts_are_connected_before_composer(tmp_path):
    llm = CaptureLlm()
    settings = load_settings(
        env={
            "DEMO_EVIDENCE_DIR": str(tmp_path / "ev"),
            "CONTEXTUALIZE_ENABLED": "false",
            "GRADER_ENABLED": "false",
            "PERSONA_PROMPTS_ENABLED": "true",
        }
    )
    ctx = build_context(
        settings,
        rag_adapter=FakeRagAdapter(result=_rag_result()),
        llm=llm,
    )

    ctx.graph.invoke(
        {
            "session_id": "s",
            "thread_id": "t",
            "input_type": "text",
            "user_input": "코퍼스에 없는 생소한 질문 zzz",
        },
        {"configurable": {"thread_id": "t"}},
    )

    assert len(llm.prompts) == 1
    assert "<persona>" in llm.prompts[0]
    assert "<response_policy>" in llm.prompts[0]
    assert "<response_examples>" in llm.prompts[0]


def test_disabled_persona_keeps_composer_prompt_without_extension(tmp_path):
    llm = CaptureLlm()
    settings = load_settings(
        env={
            "DEMO_EVIDENCE_DIR": str(tmp_path / "ev"),
            "CONTEXTUALIZE_ENABLED": "false",
            "GRADER_ENABLED": "false",
            "PERSONA_PROMPTS_ENABLED": "false",
        }
    )
    ctx = build_context(
        settings,
        rag_adapter=FakeRagAdapter(result=_rag_result()),
        llm=llm,
    )

    ctx.graph.invoke(
        {
            "session_id": "s",
            "thread_id": "t",
            "input_type": "text",
            "user_input": "코퍼스에 없는 생소한 질문 zzz",
        },
        {"configurable": {"thread_id": "t"}},
    )

    assert len(llm.prompts) == 1
    assert "<persona>" not in llm.prompts[0]
    assert "[근거]" in llm.prompts[0]

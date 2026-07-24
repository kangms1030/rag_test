"""순수 분기 함수(부작용 없음). LangGraph 조건부 엣지에서 사용.

FastAPI 에는 라우팅 로직을 두지 않는다 — 모든 라우팅은 여기서 결정된다.
"""

from __future__ import annotations

from .state import ChatState


def select_input_kind(state: ChatState) -> str:
    """버튼 입력이면 'action', 자유 입력이면 'text'."""
    return "action" if state.get("input_type") == "action" else "text"


def select_route(state: ChatState) -> str:
    """route_decider 결과에 따라 다음 노드 선택."""
    route = state.get("route")
    if route == "scenario":
        return "scenario_answer"
    return "rag3x_answer"


def select_after_eval(state: ChatState) -> str:
    """rag_result_evaluator 이후: 웹검색으로 갈지 최종으로 갈지."""
    return "web_search_answer" if state.get("route") == "web_search" else "final_formatter"


def decide_route(state: ChatState) -> tuple[str, str]:
    """route_decider 의 핵심 결정(테스트 용이하도록 분리).

    반환: (route, reason)
      - 버튼/시나리오 액션 → "scenario"
      - 유사도 매칭 채택 → "scenario"(FAQ 답변도 scenario route 로 통일)
      - 그 외 → "rag3x"
    """
    if state.get("input_type") == "action":
        return "scenario", "버튼 선택 → 시나리오 결정론 이동"

    match = state.get("scenario_match") or {}
    decision = match.get("decision")
    if decision in ("exact", "accept"):
        return "scenario", f"모범 질답 유사도 통과({decision})"
    return "rag3x", f"모범 질답 미통과({decision or 'none'}) → RAG"


def evaluate_rag_result(state: ChatState, *, web_enabled: bool, web_scope: str) -> tuple[str, str, list[str]]:
    """rag 결과가 답변 가능한지 판단하고 다음 route 결정.

    반환: (route, reason, warnings)
      route ∈ {"rag3x"(그대로 최종), "web_search", "abstain"}

    설계: rag3x CLI(ask_cli_gemini.py)는 confidence/abstain 과 무관하게 엔진이 생성한
    final_answer 를 항상 그대로 보여준다. rag3 의 abstain/low 신뢰도는 "확인 불가",
    "제공된 근거" 등 문구를 substring 매칭하는 보수적 태그라, 근거(페이지)를 인용한
    정상 답변에도 자주 붙는다(예: "제공된 근거에 따르면 …"). 따라서 실제 답변이 있으면
    폐기하지 않고 그대로 제시하되, 저신뢰일 때는 주의 문구만 덧붙인다. 진짜로 답변이
    비었을 때(answer_path=="none" 또는 빈 문자열)만 웹검색/보류로 보낸다.
    """
    warnings: list[str] = []
    rag = state.get("rag_result") or {}
    confidence = rag.get("confidence")
    answer_path = rag.get("answer_path")
    verification = rag.get("verification") or {}
    abstained = bool(verification.get("abstain"))
    final_answer = (rag.get("final_answer") or "").strip()

    has_answer = bool(final_answer) and answer_path not in ("none", None)

    if has_answer:
        low_conf = confidence in ("abstain", "low", "unknown") or abstained
        if low_conf:
            warnings.append(
                "이 답변은 내부 자료(RAG)를 근거로 생성되었으나 신뢰도가 낮게 평가되었습니다. "
                "정확한 조치는 담당 선생님이나 스쿨넷 지원센터(1899-0979) 확인을 권장합니다."
            )
            return (
                "rag3x",
                f"RAG 답변 제시(저신뢰 confidence={confidence}, path={answer_path})",
                warnings,
            )
        return "rag3x", f"RAG 답변 채택(confidence={confidence}, path={answer_path})", warnings

    # 답변 자체가 없음(빈 응답 / path=none) → 웹검색 가능 여부 판단
    if web_enabled and web_scope in ("in_domain_unresolved", "any_unresolved"):
        return "web_search", f"RAG 무응답(confidence={confidence}) → 웹검색", warnings

    warnings.append(
        "내부 자료(RAG)에서 답변을 찾지 못했고 웹검색이 비활성화되어 있어 답변을 보류합니다."
    )
    return "abstain", f"RAG 무응답(confidence={confidence}), 웹검색 비활성 → 보류", warnings

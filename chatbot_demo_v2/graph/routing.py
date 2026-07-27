"""순수 분기 함수(부작용 없음). LangGraph 조건부 엣지에서 사용.

FastAPI 에는 라우팅 로직을 두지 않는다 — 모든 라우팅은 여기서 결정된다.
Phase 3~4 에서 clarify/grader 분기가 추가된다.
"""

from __future__ import annotations

from .state import ChatState


def select_input_kind(state: ChatState) -> str:
    """버튼 입력이면 'action', 자유 입력이면 'text'."""
    return "action" if state.get("input_type") == "action" else "text"


def select_route(state: ChatState) -> str:
    """route_decider 결과에 따라 다음 노드 선택.

    scenario/faq → scenario_answer(둘 다 결정론 저장답변), clarify → clarify_node(HITL),
    그 외 → rag3x_answer.
    """
    route = state.get("route")
    if route in ("scenario", "faq"):
        return "scenario_answer"
    if route == "clarify":
        return "clarify_node"
    return "rag3x_answer"


def select_after_eval(state: ChatState) -> str:
    """rag_result_evaluator 이후 3분기.

    web_search → 웹검색 노드 / rag3x(답변 있음) → 합성 / abstain(답변 없음) → 최종.
    """
    route = state.get("route")
    if route == "web_search":
        return "web_search_answer"
    if route == "rag3x":
        return "compose_answer"
    return "final_formatter"          # abstain — 합성할 답변이 없음


def select_after_scenario_answer(state: ChatState) -> str:
    """scenario_answer 이후: FAQ 답변만 합성 경로로(시나리오 버튼 종단답변은 원문 유지)."""
    return "compose_answer" if state.get("answer_source") == "faq_match" else "final_formatter"


def select_after_grader(state: ChatState) -> str:
    """answer_grader 이후: 미해결이면 RAG 재시도(사이클), 아니면 최종."""
    return "rag3x_answer" if state.get("_escalate") else "final_formatter"


def decide_route(state: ChatState, *, clarify_enabled: bool = False,
                 clarify_min_score: float = 0.75) -> tuple[str, str]:
    """route_decider 의 핵심 결정(테스트 용이하도록 분리).

    반환: (route, reason)
      - 버튼/시나리오 액션 → "scenario"
      - 유사도 매칭 채택(exact/accept) → "faq"(결정론 저장답변)
      - 애매(reject_ambiguous) ∧ clarify 활성 ∧ 최고점수≥임계 → "clarify"(HITL 되묻기)
      - 그 외 → "rag3x"
    """
    if state.get("input_type") == "action":
        return "scenario", "버튼 선택 → 시나리오 결정론 이동"

    match = state.get("scenario_match") or {}
    decision = match.get("decision")
    if decision in ("exact", "accept"):
        return "faq", f"모범 질답 유사도 통과({decision})"
    if (clarify_enabled and decision == "reject_ambiguous"
            and (match.get("best_score") or 0.0) >= clarify_min_score):
        return "clarify", (
            f"애매 매칭(best={match.get('best_score')}, "
            f"margin={match.get('margin_observed')}) → 되묻기"
        )
    return "rag3x", f"모범 질답 미통과({decision or 'none'}) → RAG"


def select_after_clarify(state: ChatState) -> str:
    """clarify 재개 후: 사용자가 후보를 고르면 faq, '해당없음'이면 rag."""
    return "scenario_answer" if state.get("route") == "faq" else "rag3x_answer"


def evaluate_rag_result(state: ChatState, *, web_enabled: bool, web_scope: str) -> tuple[str, str, list[str]]:
    """rag 결과가 답변 가능한지 판단하고 다음 route 결정.

    반환: (route, reason, warnings)
      route ∈ {"rag3x"(그대로 최종), "web_search", "abstain"}

    설계: 실제 답변이 있으면 폐기하지 않고 그대로 제시하되, 저신뢰일 때는 주의 문구만 덧붙인다.
    진짜로 답변이 비었을 때(answer_path=="none" 또는 빈 문자열)만 웹검색/보류로 보낸다.
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

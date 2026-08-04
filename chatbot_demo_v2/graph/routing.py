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


def select_after_grader(state: ChatState, *, web_enabled: bool = False) -> str:
    """answer_grader 이후 3분기.

      FAQ 미해결(예산 남음)      → rag3x_answer   (에스컬레이션 사이클)
      RAG 미해결 ∧ 웹검색 ON     → web_search_answer  (마지막 보루)
      그 외                      → final_formatter

    2026-08-03 추가 — 실사용의 반려는 "빈 답변"보다 **"자료에서 확인할 수 없습니다" 류 답변**으로
    나타난다. 그 경우 rag_result_evaluator 는 답변이 있다고 보고 통과시키므로, 웹검색은
    grader 가 UNRESOLVED 를 낸 뒤에 붙어야 실제로 발동한다.
    """
    if state.get("_escalate"):
        return "rag3x_answer"
    if (web_enabled and state.get("route") == "rag3x"
            and state.get("grader_verdict") == "unresolved"):
        return "web_search_answer"
    return "final_formatter"


def decide_route(state: ChatState, *, clarify_enabled: bool = False,
                 clarify_min_score: float = 0.75) -> tuple[str, str]:
    """route_decider 의 핵심 결정(테스트 용이하도록 분리).

    반환: (route, reason)
      - 버튼/시나리오 액션 → "scenario"
      - 유사도 매칭 채택(exact/accept) → "faq"(결정론 저장답변)
      - 후보 **2건 모두** 하한 이상 ∧ clarify 활성 → "clarify"(HITL 되묻기)
      - 그 외 → "rag3x"
    """
    if state.get("input_type") == "action":
        return "scenario", "버튼 선택 → 시나리오 결정론 이동"

    match = state.get("scenario_match") or {}
    decision = match.get("decision")
    best = match.get("best_score") or 0.0
    margin_obs = match.get("margin_observed") or 0.0
    # 2위 점수. 정의상 best - margin 이므로 키가 없는 호출자(구 테스트)도 복원된다.
    second = match.get("second_score")
    if second is None:
        second = best - margin_obs
    if decision in ("exact", "accept"):
        return "faq", f"모범 질답 유사도 통과({decision})"

    # 2026-07-27: 되묻기 진입을 **회색지대까지** 확장한다.
    #   기존: reject_ambiguous(1·2위 차가 작을 때)일 때만 되묻었다.
    #   문제: 점수가 애매하게 낮은(자동채택 미만·완전 무관 이상) 질문은 전부 RAG 로 직행해
    #         한 가지 해석만 골라 답했다. 발단 질문 "학교에서 새로운 ap를 설치하고 싶어" 가
    #         정확히 이 구간이다(도입신청/시스템등록/물리설치 중 무엇인지 모호).
    #   변경: clarify_min_score <= best < threshold 도 되묻기로 보낸다.
    #
    # 2026-08-04: 그 회색지대가 **너무 넓어** 무관한 질문까지 되묻고 있었다. 제보 사례
    #   "무선 AP 배치 시 2.4GHz/5GHz 채널 간섭을 줄이는 채널 분배 원리는?" → best 0.642,
    #   **second 0.002**. 되묻기 화면은 "비슷한 문의가 여러 건 있어요"라고 말하면서 유사도
    #   0.00 짜리 후보를 함께 띄운다 — 후보가 사실상 1건뿐이면 되묻기가 성립하지 않는다.
    #   변경: best 뿐 아니라 **2위도 하한 이상**일 때만 되묻는다(= 진짜로 여러 해석이 경합).
    #   하한도 자동채택 임계와 같은 0.80 으로 올렸다 → 되묻기의 정의가 한 문장이 된다:
    #   **"후보 2건이 둘 다 정답급인데 어느 쪽인지 못 고를 때"**.
    #   골든셋 실측(scripts/calibrate_faq_threshold.py 와 같은 매처로 40문항 재측정):
    #     되묻기 유지  ambig_02 0.942/0.889 · ambig_03 0.971/0.964
    #                  para_07 0.985/0.968 · para_10 0.931/0.891
    #     되묻기 차단  제보사례 0.642/0.002 · rag_04 0.563/0.133 · para_04 0.351/0.000
    #                  para_08 0.928/0.644 · ambig_04 0.780/0.613
    #     대가         ambig_01 0.386/0.077 은 이제 RAG 로 간다(1건뿐인 후보라 구분 불가)
    #
    # 하한은 스케일마다 다르다 — 의미 유사도(0.80)와 문자 유사도(0.75)는 호환되지 않는다.
    # 매처가 fuzz 로 폴백하면 match["scale"] 이 "fuzz" 로 오므로 그 값을 쓴다.
    if (match.get("scale") or "fuzz") == "fuzz":
        clarify_min_score = max(clarify_min_score, 0.75)
    if clarify_enabled and best >= clarify_min_score:
        if second < clarify_min_score:
            return "rag3x", (f"후보 1건뿐(best={round(best, 3)}, "
                             f"2위={round(second, 3)} < 하한 {clarify_min_score}) → RAG")
        if decision == "reject_ambiguous":
            return "clarify", (f"애매 매칭(best={round(best, 3)}, "
                               f"margin={match.get('margin_observed')}) → 되묻기")
        if decision == "reject_low_score":
            return "clarify", (f"회색지대(best={round(best, 3)} < 임계 "
                               f"{match.get('threshold')}, 2위={round(second, 3)}) → 되묻기")
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


def parse_domain_verdict(raw: str | None) -> bool | None:
    """도메인 게이트 LLM 출력 → True(범위 안) / False(범위 밖) / None(판정 불가).

    주의: "OUT_OF_DOMAIN" 을 먼저 검사한다(부분 문자열 오탐 방지 — answer_grader 와 같은 이유).
    """
    if not raw:
        return None
    up = raw.upper()
    if "OUT_OF_DOMAIN" in up:
        return False
    if "IN_DOMAIN" in up:
        return True
    return None

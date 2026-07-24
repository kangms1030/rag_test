"""LangGraph 상태 스키마.

모든 필드는 JSON 직렬화 가능해야 한다(엔진/Lock 등 비직렬화 객체 금지 —
그것들은 AppContext 클로저로 접근).
세션 유지 필드(scenario_*)는 InMemorySaver 체크포인트로 턴 간 유지되고,
per-turn 필드는 load_or_update_session 이 매 턴 초기화한다.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class ChatState(TypedDict, total=False):
    # --- 세션 유지(턴 간 체크포인트) ---
    session_id: str
    thread_id: str
    scenario_id: Optional[str]
    current_node_id: Optional[str]
    scenario_path: list[str]
    scenario_completed: bool

    # --- 입력(per-turn) ---
    user_input: Optional[str]
    input_type: str                # "text" | "action"
    action_type: Optional[str]     # "scenario_option" | None
    action_scenario_id: Optional[str]
    action_node_id: Optional[str]
    selected_option_id: Optional[str]
    action_label: Optional[str]
    normalized_question: Optional[str]

    # --- 매칭(per-turn) ---
    scenario_match: Optional[dict]
    scenario_match_score: Optional[float]
    scenario_match_margin: Optional[float]

    # --- 라우팅(per-turn) ---
    route: Optional[str]           # "scenario" | "rag3x" | "web_search" | "abstain"
    route_reason: Optional[str]

    # --- 결과(per-turn) ---
    rag_result: Optional[dict]
    web_result: Optional[dict]
    final_answer: Optional[str]
    confidence: Optional[str]      # "high"|"low"|"abstain"|"unknown"|"n/a"
    answer_path: Optional[str]     # "scenario"|"text"|"vision"|"web"|"none"
    answer_source: Optional[str]   # "scenario_tree"|"faq_match"|"rag3x"|"web"|"none"
    options: list[dict]
    evidence: list[dict]
    verification: Optional[dict]
    source_meta: Optional[dict]    # 시트/행/질문유형/장애유형/근거파일명 등

    # --- 관측/디버그 ---
    trace: list[dict]
    timings: dict
    errors: list[str]
    warnings: list[str]

    # 내부 제어 플래그
    _turn_started_at: float
    _rag_run_id: Optional[str]


def new_turn_defaults() -> dict:
    """per-turn 필드 초기값(세션 유지 필드는 건드리지 않음)."""
    return {
        "normalized_question": None,
        "scenario_match": None,
        "scenario_match_score": None,
        "scenario_match_margin": None,
        "route": None,
        "route_reason": None,
        "rag_result": None,
        "web_result": None,
        "final_answer": None,
        "confidence": None,
        "answer_path": None,
        "answer_source": None,
        "options": [],
        "evidence": [],
        "verification": None,
        "source_meta": None,
        "trace": [],
        "timings": {},
        "errors": [],
        "warnings": [],
        "_rag_run_id": None,
    }

"""LangGraph 상태 스키마 (v2).

ChatState(메인그래프) + RagState(RAG 서브그래프).
모든 필드는 JSON 직렬화 가능해야 한다(엔진/Lock/Config 등 비직렬화 객체 금지 — ctx 클로저로 접근).
세션 유지 필드(scenario_*, messages)는 InMemorySaver 체크포인트로 턴 간 유지되고,
per-turn 필드는 new_turn_defaults()/load_or_update_session 이 매 턴 초기화한다.

v1(chatbot_demo) 대비 추가:
- messages: 대화 메모리(add_messages) — 자유입력 후속질문·컨텍스트화에 사용(Phase 3).
- standalone_question/clarify_*/composed/original_answer/grader_verdict/escalate_budget/faq_evidence.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class ChatState(TypedDict, total=False):
    # --- 세션 유지(턴 간 체크포인트) ---
    session_id: str
    thread_id: str
    scenario_id: Optional[str]
    current_node_id: Optional[str]
    scenario_path: list[str]
    scenario_completed: bool
    messages: Annotated[list, add_messages]   # 대화 이력(Human/AI) — 절대 매 턴 초기화 금지

    # --- 입력(per-turn) ---
    user_input: Optional[str]
    input_type: str                # "text" | "action"
    action_type: Optional[str]     # "scenario_option" | None
    action_scenario_id: Optional[str]
    action_node_id: Optional[str]
    selected_option_id: Optional[str]
    action_label: Optional[str]
    normalized_question: Optional[str]
    standalone_question: Optional[str]   # 이력 반영 재작성(Phase 3). 없으면 user_input 사용
    contextualized: bool

    # --- 매칭(per-turn) ---
    scenario_match: Optional[dict]
    scenario_match_score: Optional[float]
    scenario_match_margin: Optional[float]

    # --- clarify(HITL, per-turn) ---
    clarify_candidates: list[dict]       # [{faq_id, question, score}]
    clarify_choice: Optional[str]        # resume 값: faq_id | "__none__"

    # --- 라우팅(per-turn) ---
    route: Optional[str]           # "scenario" | "faq" | "clarify" | "rag3x" | "web_search" | "abstain"
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
    faq_evidence: list[dict]       # FAQ 근거 페이지 이미지(Phase 4)
    verification: Optional[dict]
    source_meta: Optional[dict]

    # --- composer / grader(per-turn) ---
    composed: bool
    composed_answer: Optional[str]   # 합성 결과(채택된 경우만). final_formatter 가 우선 적용
    composer_fallback: Optional[str]
    original_answer: Optional[str]
    grader_verdict: Optional[str]  # "resolved" | "unresolved" | None
    escalate_budget: int
    _escalate: bool                # grader 가 FAQ→RAG 재시도로 보낼지(조건부 엣지에서 사용)

    # --- 관측/디버그 ---
    trace: list[dict]
    timings: dict
    errors: list[str]
    warnings: list[str]

    # 내부 제어 플래그
    _turn_started_at: float
    _rag_run_id: Optional[str]


def new_turn_defaults() -> dict:
    """per-turn 필드 초기값(세션 유지 필드 scenario_*/messages 는 건드리지 않음)."""
    return {
        "normalized_question": None,
        "standalone_question": None,
        "contextualized": False,
        "scenario_match": None,
        "scenario_match_score": None,
        "scenario_match_margin": None,
        "clarify_candidates": [],
        "clarify_choice": None,
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
        "faq_evidence": [],
        "verification": None,
        "source_meta": None,
        "composed": False,
        "composed_answer": None,
        "composer_fallback": None,
        "original_answer": None,
        "grader_verdict": None,
        "escalate_budget": 1,
        "_escalate": False,
        "trace": [],
        "timings": {},
        "errors": [],
        "warnings": [],
        "_rag_run_id": None,
    }


class RagState(TypedDict, total=False):
    """RAG 서브그래프 전용 상태. controller_x S0~S8 을 노드로 편성.

    직렬화 불가 객체(RetrievalResult 인스턴스/Backend/Config)는 넣지 않는다 —
    RetrievalResult 는 dict 로 변환해 보관, Backend/Config 는 ctx 클로저로 접근.
    """
    question: str
    run_id: str

    # S1-5 검색 (RetrievalResult 를 dict 로)
    retrieval: Optional[dict]      # {answer_path, route_reason, rerank_top_score,
                                   #  selected_pages(원본, 경로 포함), selected_documents}
    # 사이클 예산 (controller_x budget dict 승격)
    crag_budget: int
    answer_regen_budget: int
    path_switch_budget: int
    rewritten_query: Optional[str]

    # S6-7
    answer: Optional[dict]         # {final_answer, raw, context, transcription?}
    answer_path: str               # "text" | "vision" | "none"
    verify: Optional[dict]

    # 제어
    deadline_ts: float
    history: Annotated[list[dict], operator.add]

    # 출력
    result: Optional[dict]         # _finalize 산출물(원본 ask()와 동일 스키마)

"""LangGraph 그래프 구성 및 컴파일 (v2).

Phase 1: v1 동등 메인그래프(전진 DAG). rag3x_answer 는 blackbox.
Phase 2 에서 build_rag_subgraph 추가 + rag3x_answer 교체.
Phase 3~4 에서 contextualize/clarify/compose/grader 노드·엣지 추가.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import make_nodes
from .rag_nodes import RagDeps, make_rag_nodes
from .routing import (
    select_after_eval,
    select_after_grader,
    select_after_scenario_answer,
    select_input_kind,
    select_route,
)
from .state import ChatState, RagState


def _metrics_wrap(deps: RagDeps, fn):
    """노드 본문을 with metrics.run_metrics(m) 로 감싼다(m=현재 run 의 RunMetrics).

    langgraph 가 노드를 다른 스레드로 실행하더라도 rag3 원시함수의 record_* 가 누산되도록
    (contextvar 전파 미보장 방어) 노드마다 명시적으로 컨텍스트를 진입한다.
    """
    def wrapped(state):
        m = deps.scratch.get("metrics")
        if m is None:
            return fn(state)
        from rag3 import metrics
        with metrics.run_metrics(m):
            return fn(state)
    return wrapped


def build_rag_subgraph(deps: RagDeps):
    """controller_x S0~S8 을 편성한 RAG 서브그래프를 컴파일해 반환.

    체크포인터는 두지 않는다(단발 실행 — 부모 그래프가 대화 상태를 관리).
    """
    nodes, routers = make_rag_nodes(deps)
    g = StateGraph(RagState)

    # 외부 백엔드(Ollama 임베딩 · Gemini)를 타는 노드만 일시 오류 재시도.
    # 그 외(prepare/finalize/rollback 판단)는 순수 로직이라 재시도 의미가 없다.
    # 주의: 메인그래프의 LLM 노드(compose/grader)는 LlmHelper 가 예외를 삼키고 None 을
    # 반환해 pass-through 하므로 RetryPolicy 가 발동하지 않는다(그래서 붙이지 않음).
    # Gemini 자체의 429/5xx 지수백오프는 GeminiBackend 안에 이미 있다.
    _retry_nodes = {"retrieve", "grade_evidence", "answer_node", "verify_node"}
    try:
        from langgraph.types import RetryPolicy

        retry = RetryPolicy(max_attempts=2)
    except Exception:  # noqa: BLE001 - 구버전 호환
        retry = None

    for name, fn in nodes.items():
        wrapped = _metrics_wrap(deps, fn)
        if retry is not None and name in _retry_nodes:
            g.add_node(name, wrapped, retry_policy=retry)
        else:
            g.add_node(name, wrapped)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "retrieve")
    # 2026-07-27: retrieve 와 answer 사이에 grade_evidence(근거 3등급 판정)를 넣는다.
    # 검색 랭킹만으로는 못 고치는 실패를 의미 판정으로 잡고, "의미상 근거가 없을 때"
    # CRAG 재질의가 발동하게 한다(기존에는 리랭크 점수가 바닥일 때만 발동).
    g.add_conditional_edges(
        "retrieve", routers["after_retrieve"],
        {"crag": "crag_rewrite", "no_answer": "finalize", "grade": "grade_evidence"},
    )
    g.add_conditional_edges(
        "grade_evidence", routers["after_grade"],
        {"crag": "crag_rewrite", "no_answer": "finalize", "answer": "answer_node"},
    )
    g.add_edge("crag_rewrite", "retrieve")            # ← 사이클(1회, crag_budget 게이트)
    g.add_edge("answer_node", "verify_node")
    g.add_conditional_edges(
        "verify_node", routers["after_verify"],
        {
            "rollback_top1": "rollback_top1",
            "rollback_vision": "rollback_vision",
            "rollback_ocr": "rollback_ocr",
            "done": "finalize",
        },
    )
    g.add_edge("rollback_top1", "finalize")
    g.add_edge("rollback_vision", "finalize")
    g.add_edge("rollback_ocr", "finalize")
    g.add_edge("finalize", END)
    # name 을 주지 않으면 LangSmith 에 자식 run 이 "LangGraph" 로 찍혀 어떤 서브그래프인지
    # 구분되지 않는다(실측 확인) → 트레이스 가독성을 위해 명시.
    return g.compile(name="rag_subgraph")


def build_graph(ctx: Any, checkpointer: Any | None = None):
    """AppContext 로 노드를 구성하고 컴파일된 그래프를 반환한다.

    checkpointer 미지정 시 InMemorySaver 사용(실험용 — 재시작 시 대화 초기화).
    """
    nodes = make_nodes(ctx)
    g = StateGraph(ChatState)

    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, "normalize_input")
    g.add_edge("normalize_input", "load_or_update_session")

    # 버튼/자유 입력 분기 (자유입력은 contextualize 를 먼저 거친다)
    g.add_conditional_edges(
        "load_or_update_session",
        select_input_kind,
        {"action": "scenario_action_handler", "text": "contextualize_query"},
    )
    g.add_edge("scenario_action_handler", "route_decider")
    g.add_edge("contextualize_query", "scenario_matcher")
    g.add_edge("scenario_matcher", "route_decider")

    # scenario/faq → scenario_answer, clarify → clarify_node(HITL), rag → rag3x_answer
    g.add_conditional_edges(
        "route_decider",
        select_route,
        {
            "scenario_answer": "scenario_answer",
            "clarify_node": "clarify_node",
            "rag3x_answer": "rag3x_answer",
        },
    )
    # clarify_node 는 Command(goto=...) 로 재개 후 분기 → 정적 엣지 불필요.
    # FAQ 답변만 합성 경로로, 시나리오 버튼 종단답변은 원문 그대로 최종으로.
    g.add_conditional_edges(
        "scenario_answer",
        select_after_scenario_answer,
        {"compose_answer": "compose_answer", "final_formatter": "final_formatter"},
    )

    # rag 평가 후 web/합성/최종(abstain) 3분기
    g.add_edge("rag3x_answer", "rag_result_evaluator")
    g.add_conditional_edges(
        "rag_result_evaluator",
        select_after_eval,
        {
            "web_search_answer": "web_search_answer",
            "compose_answer": "compose_answer",
            "final_formatter": "final_formatter",
        },
    )
    g.add_edge("web_search_answer", "final_formatter")

    # 합성 → 해결도 판정 → (FAQ 미해결이면 RAG 재시도 사이클 / RAG 미해결이면 웹검색) → 최종
    g.add_edge("compose_answer", "answer_grader")
    # 웹검색 활성 여부는 설정 고정값이라 컴파일 시점에 닫아 둔다(런타임 상태에 넣지 않는다).
    web_enabled = bool(getattr(getattr(ctx, "settings", None), "web_search_enabled", False))

    def _after_grader(state):
        return select_after_grader(state, web_enabled=web_enabled)

    g.add_conditional_edges(
        "answer_grader",
        _after_grader,
        {
            "rag3x_answer": "rag3x_answer",
            "web_search_answer": "web_search_answer",
            "final_formatter": "final_formatter",
        },
    )
    g.add_edge("final_formatter", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)

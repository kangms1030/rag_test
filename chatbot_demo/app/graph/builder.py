"""LangGraph 그래프 구성 및 컴파일."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import make_nodes
from .routing import select_after_eval, select_input_kind, select_route
from .state import ChatState


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

    # 버튼/자유 입력 분기
    g.add_conditional_edges(
        "load_or_update_session",
        select_input_kind,
        {"action": "scenario_action_handler", "text": "scenario_matcher"},
    )
    g.add_edge("scenario_action_handler", "route_decider")
    g.add_edge("scenario_matcher", "route_decider")

    # scenario vs rag 분기
    g.add_conditional_edges(
        "route_decider",
        select_route,
        {"scenario_answer": "scenario_answer", "rag3x_answer": "rag3x_answer"},
    )
    g.add_edge("scenario_answer", "final_formatter")

    # rag 평가 후 web/최종 분기
    g.add_edge("rag3x_answer", "rag_result_evaluator")
    g.add_conditional_edges(
        "rag_result_evaluator",
        select_after_eval,
        {"web_search_answer": "web_search_answer", "final_formatter": "final_formatter"},
    )
    g.add_edge("web_search_answer", "final_formatter")
    g.add_edge("final_formatter", END)

    if checkpointer is None:
        checkpointer = InMemorySaver()
    return g.compile(checkpointer=checkpointer)

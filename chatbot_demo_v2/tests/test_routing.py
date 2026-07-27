"""순수 라우팅 함수 검증."""

from __future__ import annotations

from chatbot_demo_v2.graph.routing import (
    decide_route,
    evaluate_rag_result,
    select_after_clarify,
    select_after_eval,
    select_input_kind,
    select_route,
)


def test_select_input_kind():
    assert select_input_kind({"input_type": "action"}) == "action"
    assert select_input_kind({"input_type": "text"}) == "text"


def test_decide_route_action_is_scenario():
    route, _ = decide_route({"input_type": "action"})
    assert route == "scenario"


def test_decide_route_accepted_match_is_faq():
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "accept"}}
    )
    assert route == "faq"
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "exact"}}
    )
    assert route == "faq"


def test_decide_route_low_score_is_rag():
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "reject_low_score"}}
    )
    assert route == "rag3x"


def test_decide_route_ambiguous_no_clarify_is_rag():
    # clarify 비활성(기본) → 애매도 rag 로.
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "reject_ambiguous", "best_score": 0.85}}
    )
    assert route == "rag3x"


def test_decide_route_ambiguous_high_score_is_clarify():
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "reject_ambiguous", "best_score": 0.85}},
        clarify_enabled=True, clarify_min_score=0.75,
    )
    assert route == "clarify"


def test_decide_route_ambiguous_low_score_no_clarify():
    # 애매하지만 최고점수가 임계 미만 → clarify 안 하고 rag.
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "reject_ambiguous", "best_score": 0.60}},
        clarify_enabled=True, clarify_min_score=0.75,
    )
    assert route == "rag3x"


def test_select_route_faq_and_clarify():
    assert select_route({"route": "faq"}) == "scenario_answer"
    assert select_route({"route": "clarify"}) == "clarify_node"


def test_select_after_clarify():
    assert select_after_clarify({"route": "faq"}) == "scenario_answer"
    assert select_after_clarify({"route": "rag3x"}) == "rag3x_answer"


def test_evaluate_rag_high_confidence_stays():
    state = {"rag_result": {"confidence": "high", "answer_path": "text", "final_answer": "ok"}}
    route, _, warns = evaluate_rag_result(state, web_enabled=False, web_scope="in_domain_unresolved")
    assert route == "rag3x"
    assert warns == []


def test_evaluate_rag_abstain_tag_but_has_answer_is_shown():
    state = {
        "rag_result": {
            "confidence": "abstain",
            "answer_path": "text",
            "final_answer": "제공된 근거에 따르면 PC 1대만 안 되는 경우입니다. 42~45쪽을 참고하세요.",
            "verification": {"abstain": True},
        }
    }
    route, _, warns = evaluate_rag_result(state, web_enabled=False, web_scope="in_domain_unresolved")
    assert route == "rag3x"
    assert warns


def test_evaluate_rag_low_confidence_with_answer_is_shown():
    state = {"rag_result": {"confidence": "low", "answer_path": "vision", "final_answer": "해결 방법은 …"}}
    route, _, warns = evaluate_rag_result(state, web_enabled=False, web_scope="in_domain_unresolved")
    assert route == "rag3x"
    assert warns


def test_evaluate_rag_abstain_web_disabled_goes_abstain():
    state = {"rag_result": {"confidence": "abstain", "answer_path": "none", "final_answer": ""}}
    route, _, warns = evaluate_rag_result(state, web_enabled=False, web_scope="in_domain_unresolved")
    assert route == "abstain"
    assert warns


def test_evaluate_rag_abstain_web_enabled_goes_web():
    state = {"rag_result": {"confidence": "unknown", "answer_path": "none", "final_answer": ""}}
    route, _, _ = evaluate_rag_result(state, web_enabled=True, web_scope="in_domain_unresolved")
    assert route == "web_search"


def test_select_after_eval():
    assert select_after_eval({"route": "web_search"}) == "web_search_answer"
    assert select_after_eval({"route": "abstain"}) == "final_formatter"   # 합성할 답변 없음
    assert select_after_eval({"route": "rag3x"}) == "compose_answer"      # 답변 있음 → 합성


def test_select_route():
    assert select_route({"route": "scenario"}) == "scenario_answer"
    assert select_route({"route": "rag3x"}) == "rag3x_answer"

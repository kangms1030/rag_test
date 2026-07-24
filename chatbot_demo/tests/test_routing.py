"""순수 라우팅 함수 검증."""

from __future__ import annotations

from chatbot_demo.app.graph.routing import (
    decide_route,
    evaluate_rag_result,
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


def test_decide_route_accepted_match_is_scenario():
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "accept"}}
    )
    assert route == "scenario"
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "exact"}}
    )
    assert route == "scenario"


def test_decide_route_low_score_is_rag():
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "reject_low_score"}}
    )
    assert route == "rag3x"


def test_decide_route_ambiguous_is_rag():
    route, _ = decide_route(
        {"input_type": "text", "scenario_match": {"decision": "reject_ambiguous"}}
    )
    assert route == "rag3x"


def test_evaluate_rag_high_confidence_stays():
    state = {"rag_result": {"confidence": "high", "answer_path": "text", "final_answer": "ok"}}
    route, _, warns = evaluate_rag_result(state, web_enabled=False, web_scope="in_domain_unresolved")
    assert route == "rag3x"
    assert warns == []


def test_evaluate_rag_abstain_tag_but_has_answer_is_shown():
    # rag3 는 "제공된 근거에 따르면 …" 같은 정상 답변에도 abstain 태그를 붙인다.
    # 실제 답변이 있으면 CLI 처럼 그대로 제시하되 저신뢰 경고만 덧붙인다.
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
    assert warns  # 저신뢰 주의 문구


def test_evaluate_rag_low_confidence_with_answer_is_shown():
    state = {"rag_result": {"confidence": "low", "answer_path": "vision", "final_answer": "해결 방법은 …"}}
    route, _, warns = evaluate_rag_result(state, web_enabled=False, web_scope="in_domain_unresolved")
    assert route == "rag3x"
    assert warns


def test_evaluate_rag_abstain_web_disabled_goes_abstain():
    state = {"rag_result": {"confidence": "abstain", "answer_path": "none", "final_answer": ""}}
    route, _, warns = evaluate_rag_result(state, web_enabled=False, web_scope="in_domain_unresolved")
    assert route == "abstain"
    assert warns  # 경고 존재


def test_evaluate_rag_abstain_web_enabled_goes_web():
    state = {"rag_result": {"confidence": "unknown", "answer_path": "none", "final_answer": ""}}
    route, _, _ = evaluate_rag_result(state, web_enabled=True, web_scope="in_domain_unresolved")
    assert route == "web_search"


def test_select_after_eval():
    assert select_after_eval({"route": "web_search"}) == "web_search_answer"
    assert select_after_eval({"route": "abstain"}) == "final_formatter"
    assert select_after_eval({"route": "rag3x"}) == "final_formatter"


def test_select_route():
    assert select_route({"route": "scenario"}) == "scenario_answer"
    assert select_route({"route": "rag3x"}) == "rag3x_answer"

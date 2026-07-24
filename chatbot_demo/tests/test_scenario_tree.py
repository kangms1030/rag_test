"""시나리오 트리 로딩/네비게이션/검증."""

from __future__ import annotations

import pytest

from chatbot_demo.scenario.tree import InvalidActionError


def test_root_has_nine_options(tree):
    assert len(tree.root().options) == 9


def test_deterministic_navigation(tree):
    # root -> 인터넷이 안 돼요 -> 학교 전체 -> 15분 이상 (terminal)
    n1 = tree.resolve_option("root", "internet_down")
    assert n1.node_id == "internet_down.situation"
    n2 = tree.resolve_option(n1.node_id, "school_all")
    assert n2.node_id == "internet_down.all.duration"
    n3 = tree.resolve_option(n2.node_id, "over_15")
    assert n3.is_terminal
    assert n3.answer_text.startswith("15분 이상")


def test_shared_terminal_for_two_menu_buttons(tree):
    a = tree.resolve_option("root", "nms_no_account")
    b = tree.resolve_option("root", "nms_lost_account")
    assert a.node_id == b.node_id == "nms_account.answer"
    assert "1588-5509" in a.answer_text


def test_restart_option_returns_to_root(tree):
    term = tree.resolve_option("root", "callcenter")
    root = tree.resolve_option(term.node_id, "__restart__")
    assert root.node_id == tree.root_node_id


def test_invalid_option_raises(tree):
    with pytest.raises(InvalidActionError):
        tree.resolve_option("root", "does_not_exist")


def test_invalid_node_raises(tree):
    with pytest.raises(InvalidActionError):
        tree.get_node("no_such_node")


def test_all_terminals_have_answers(tree):
    for node in tree.nodes.values():
        if node.is_terminal:
            assert node.answer_text, f"{node.node_id} 답변 없음"
            assert node.answer_source == "scenario_ppt"

"""LangGraph 노드 클로저(make_nodes(ctx)).

각 노드는 ChatState 부분 업데이트(dict)를 반환한다. 비직렬화 객체(엔진/락)는
상태에 넣지 않고 ctx 를 통해 접근한다. 모든 라우팅 판단은 여기(+routing.py)에서 이뤄진다.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ...scenario.matcher import normalize_text
from ...scenario.tree import InvalidActionError
from ..graph.routing import decide_route, evaluate_rag_result
from ..graph.state import ChatState, new_turn_defaults

ABSTAIN_MESSAGE = (
    "죄송합니다. 현재 내부 자료로는 정확한 답변을 드리기 어렵습니다. "
    "학교 정보부 담당 선생님께 문의하시거나, 스쿨넷 서비스 지원센터(1899-0979)로 "
    "연락해 주세요."
)


class EmptyInputError(Exception):
    """빈 자유 입력."""


def _node_meta(meta: dict, tags: list | None = None) -> None:
    """현재 노드 run 에 판단 근거 metadata(+tags)를 붙인다(추적 켜져 있을 때만)."""
    try:
        from ...observability.langsmith import add_node_metadata

        add_node_metadata(meta, tags=tags)
    except Exception:
        pass


def _trace(state: ChatState, node: str, detail: str) -> dict:
    rows = list(state.get("trace") or [])
    rows.append({"node": node, "detail": detail})
    return rows


def make_nodes(ctx: Any) -> dict[str, Callable[[ChatState], dict]]:
    tree = ctx.tree
    faq = ctx.faq
    matcher = ctx.matcher
    settings = ctx.settings

    # ---------- 1. normalize_input ----------
    def normalize_input(state: ChatState) -> dict:
        out = new_turn_defaults()
        out["_turn_started_at"] = time.time()
        input_type = state.get("input_type") or "text"

        if input_type == "action":
            # 액션 필드 검증
            if state.get("action_type") != "scenario_option":
                raise InvalidActionError("지원하지 않는 action.type")
            if not state.get("action_node_id") or not state.get("selected_option_id"):
                raise InvalidActionError("action 에 node_id/option_id 누락")
            out["normalized_question"] = None
            out["trace"] = _trace(
                {"trace": out["trace"]},
                "normalize_input",
                f"버튼 입력(node={state.get('action_node_id')}, "
                f"option={state.get('selected_option_id')})",
            )
        else:
            raw = (state.get("user_input") or "").strip()
            if not raw:
                raise EmptyInputError("빈 질문")
            out["normalized_question"] = normalize_text(raw)
            out["trace"] = _trace(
                {"trace": out["trace"]},
                "normalize_input",
                f"자유 입력 정규화(len={len(raw)})",
            )
        return out

    # ---------- 2. load_or_update_session ----------
    def load_or_update_session(state: ChatState) -> dict:
        out: dict = {}
        if state.get("input_type") == "text":
            # 자유 입력: 이전 시나리오 위치가 라우팅에 영향 주지 않도록 중단.
            out.update(
                scenario_id=None,
                current_node_id=None,
                scenario_completed=False,
                scenario_path=[],
            )
            detail = "자유 입력 → 시나리오 상태 중단"
        else:
            # 버튼 입력: 체크포인트의 시나리오 상태 유지(핸들러가 갱신).
            if state.get("scenario_path") is None:
                out["scenario_path"] = []
            detail = "버튼 입력 → 시나리오 상태 유지"
        out["trace"] = _trace(state, "load_or_update_session", detail)
        return out

    # ---------- 3. scenario_action_handler ----------
    def scenario_action_handler(state: ChatState) -> dict:
        node_id = state.get("action_node_id")
        option_id = state.get("selected_option_id")
        label = state.get("action_label") or ""
        # 결정론적 이동(유사도 검색 없음). 실패 시 InvalidActionError → API 400.
        next_node = tree.resolve_option(node_id, option_id)

        path = list(state.get("scenario_path") or [])
        if option_id == "__restart__" or next_node.node_id == tree.root_node_id:
            path = []
            scenario_completed = False
        else:
            if label:
                path.append(label)
            scenario_completed = next_node.is_terminal

        _node_meta(
            {
                "from_node": node_id,
                "option_id": option_id,
                "to_node": next_node.node_id,
                "to_type": next_node.node_type,
                "terminal": next_node.is_terminal,
            },
            tags=["scenario_nav"],
        )
        out = {
            "scenario_id": next_node.scenario_id,
            "current_node_id": next_node.node_id,
            "scenario_path": path,
            "scenario_completed": scenario_completed,
            "trace": _trace(
                state,
                "scenario_action_handler",
                f"이동 → {next_node.node_id}({next_node.node_type})",
            ),
        }
        return out

    # ---------- 4. scenario_matcher ----------
    def scenario_matcher(state: ChatState) -> dict:
        norm = state.get("normalized_question") or ""
        mr = matcher.match(norm)
        _node_meta(
            {
                "match_decision": mr.decision,
                "best_score": round(mr.best_score, 4),
                "second_score": round(mr.second_score, 4),
                "margin_observed": round(mr.margin_observed, 4),
                "threshold": mr.threshold,
                "matched_id": mr.matched_id,
                "matched_question": mr.matched_question,
            },
            tags=[f"match:{mr.decision}"],
        )
        return {
            "scenario_match": mr.to_dict(),
            "scenario_match_score": mr.best_score,
            "scenario_match_margin": mr.margin_observed,
            "trace": _trace(
                state,
                "scenario_matcher",
                f"decision={mr.decision}, best={mr.best_score:.3f}, "
                f"margin={mr.margin_observed:.3f}",
            ),
        }

    # ---------- 5. route_decider ----------
    def route_decider(state: ChatState) -> dict:
        route, reason = decide_route(state)
        _node_meta({"route": route, "route_reason": reason}, tags=[f"route:{route}"])
        return {
            "route": route,
            "route_reason": reason,
            "trace": _trace(state, "route_decider", f"route={route} ({reason})"),
        }

    # ---------- 6. scenario_answer ----------
    def scenario_answer(state: ChatState) -> dict:
        if state.get("input_type") == "action":
            node = tree.get_node(state.get("current_node_id"))
            if node.is_terminal:
                out = {
                    "final_answer": node.answer_text,
                    "answer_path": "scenario",
                    "answer_source": "scenario_tree",
                    "confidence": "n/a",
                    "options": tree.options_payload(node),
                    "scenario_completed": True,
                    "source_meta": {
                        "type": "scenario",
                        "scenario_id": node.scenario_id,
                        "node_id": node.node_id,
                        "answer_source": node.answer_source,
                        "ref_sheet": node.answer_ref_sheet,
                        "ref_row": node.answer_ref_row,
                    },
                }
            else:
                out = {
                    "final_answer": node.text,
                    "answer_path": "scenario",
                    "answer_source": "scenario_tree",
                    "confidence": "n/a",
                    "options": tree.options_payload(node),
                    "scenario_completed": False,
                    "source_meta": {
                        "type": "scenario",
                        "scenario_id": node.scenario_id,
                        "node_id": node.node_id,
                    },
                }
            out["trace"] = _trace(
                state, "scenario_answer", f"시나리오 노드 {node.node_id} 응답"
            )
            return out

        # 자유 입력 + FAQ 유사도 통과 → 저장된 모범 답변 그대로(LLM 미사용)
        match = state.get("scenario_match") or {}
        entry = faq.get(match.get("matched_id")) if match.get("matched_id") else None
        if entry is None:
            # 방어적: 여기 오면 안 됨(route_decider 가 accept 일 때만 진입)
            return {
                "final_answer": None,
                "answer_path": "none",
                "answer_source": "none",
                "confidence": "unknown",
                "trace": _trace(state, "scenario_answer", "FAQ 매칭 항목 없음(이상)"),
            }
        return {
            "final_answer": entry.answer,  # 원문 그대로
            "answer_path": "scenario",
            "answer_source": "faq_match",
            "confidence": "n/a",
            "options": [],
            "source_meta": {
                "type": "faq",
                "sheet": entry.sheet,
                "row": entry.row,
                "no": entry.no,
                "question_type": entry.question_type,
                "fault_type": entry.fault_type,
                "source_files": entry.source_files,
                "matched_question": entry.question,
                "best_score": match.get("best_score"),
            },
            "trace": _trace(
                state, "scenario_answer",
                f"FAQ 모범답변 반환({entry.id}, score={match.get('best_score')})",
            ),
        }

    # ---------- 7. rag3x_answer ----------
    def rag3x_answer(state: ChatState) -> dict:
        question = (state.get("user_input") or "").strip()
        t0 = time.time()
        # rag3x 호출을 별도 traceable child run("rag3x.ask")으로 노출 →
        # LangSmith 트리에서 검색·리랭크·검증 결과가 rag3x 노드 하위에 보인다.
        # RagBusyError / RagUnavailableError 는 그래프 밖으로 전파 → API 429/503
        from ...observability.langsmith import traced_call

        def _do_ask(*, q):
            # 어댑터가 이미 절대경로를 제거한 정규화 결과를 반환(evidence 는 상대 URL).
            # 이 반환값이 rag3x.ask run 의 outputs 로 기록된다.
            return ctx.rag_adapter.ask(q)

        result = traced_call("rag3x.ask", _do_ask, run_type="tool", q=question)
        elapsed = time.time() - t0
        timings = dict(state.get("timings") or {})
        timings["rag_s"] = elapsed

        m = result.get("metrics") or {}
        ts = m.get("timings_seconds") or {}
        _node_meta(
            {
                "confidence": result.get("confidence"),
                "answer_path": result.get("answer_path"),
                "rerank_top_score": result.get("rerank_top_score"),
                "rag_route_reason": result.get("route_reason"),
                "evidence_count": len(result.get("evidence") or []),
                "retrieve_s": ts.get("retrieve"),
                "answer_s": ts.get("answer"),
                "total_model_calls": m.get("total_model_calls"),
                "run_id": result.get("run_id"),
            },
            tags=[f"rag_conf:{result.get('confidence')}", f"rag_path:{result.get('answer_path')}"],
        )
        return {
            "rag_result": result,
            "_rag_run_id": result.get("run_id"),
            "timings": timings,
            "trace": _trace(
                state, "rag3x_answer",
                f"RAG 응답(run={result.get('run_id')}, "
                f"conf={result.get('confidence')}, path={result.get('answer_path')})",
            ),
        }

    # ---------- 8. rag_result_evaluator ----------
    def rag_result_evaluator(state: ChatState) -> dict:
        route, reason, warns = evaluate_rag_result(
            state,
            web_enabled=settings.web_search_enabled,
            web_scope=settings.web_search_scope,
        )
        warnings = list(state.get("warnings") or []) + warns
        _node_meta({"eval_route": route, "eval_reason": reason}, tags=[f"eval:{route}"])
        return {
            "route": route,
            "route_reason": reason,
            "warnings": warnings,
            "trace": _trace(state, "rag_result_evaluator", f"route={route} ({reason})"),
        }

    # ---------- 9. web_search_answer ----------
    def web_search_answer(state: ChatState) -> dict:
        question = (state.get("user_input") or "").strip()
        t0 = time.time()
        res = ctx.web_provider.search_and_answer(question, context={})
        timings = dict(state.get("timings") or {})
        timings["web_s"] = time.time() - t0
        return {
            "web_result": res,
            "timings": timings,
            "trace": _trace(
                state, "web_search_answer",
                f"web provider={res.get('provider')}, enabled={res.get('enabled')}",
            ),
        }

    # ---------- 10. final_formatter ----------
    def final_formatter(state: ChatState) -> dict:
        route = state.get("route")
        out: dict = {}

        if route == "scenario":
            # scenario_answer 가 이미 채움 — 그대로 유지.
            out["evidence"] = []
            out["verification"] = None
        elif route == "rag3x":
            rag = state.get("rag_result") or {}
            out.update(
                final_answer=rag.get("final_answer"),
                answer_path=rag.get("answer_path") or "none",
                answer_source="rag3x",
                confidence=rag.get("confidence") or "unknown",
                evidence=rag.get("evidence") or [],
                verification=rag.get("verification"),
                options=[],
                source_meta={
                    "type": "rag3x",
                    "run_id": rag.get("run_id"),
                    "rerank_top_score": rag.get("rerank_top_score"),
                    "route_reason": rag.get("route_reason"),
                    "selected_pages": rag.get("selected_pages"),
                    "metrics": rag.get("metrics"),
                },
            )
        elif route == "web_search":
            web = state.get("web_result") or {}
            out.update(
                final_answer=web.get("answer") or "",
                answer_path="web",
                answer_source="web",
                confidence="unknown",
                evidence=[],
                verification=None,
                options=[],
                source_meta={
                    "type": "web",
                    "provider": web.get("provider"),
                    "sources": web.get("sources"),
                    "note": web.get("note"),
                },
            )
        else:  # abstain
            out.update(
                final_answer=ABSTAIN_MESSAGE,
                answer_path="none",
                answer_source="none",
                confidence="abstain",
                evidence=[],
                verification=None,
                options=[],
                source_meta={"type": "abstain"},
            )

        # 타이밍 마감
        timings = dict(state.get("timings") or {})
        started = state.get("_turn_started_at") or time.time()
        total = time.time() - started
        timings["total_s"] = total
        out["timings"] = timings

        out["trace"] = _trace(state, "final_formatter", f"route={route}, total={total:.3f}s")

        # LangSmith 동적 metadata(가능할 때만 no-op)
        try:
            from ...observability.langsmith import attach_run_metadata

            attach_run_metadata(
                {
                    "session_id": state.get("session_id"),
                    "scenario_id": state.get("scenario_id"),
                    "current_node_id": state.get("current_node_id"),
                    "route": route,
                    "route_reason": state.get("route_reason"),
                    "answer_source": out.get("answer_source") or state.get("answer_source"),
                    "scenario_match_score": state.get("scenario_match_score"),
                    "confidence": out.get("confidence") or state.get("confidence"),
                    "answer_path": out.get("answer_path") or state.get("answer_path"),
                    "elapsed_seconds": round(total, 3),
                    "rag_run_id": state.get("_rag_run_id"),
                },
                tags=[f"turn_route:{route}"],
            )
        except Exception:
            pass

        return out

    return {
        "normalize_input": normalize_input,
        "load_or_update_session": load_or_update_session,
        "scenario_action_handler": scenario_action_handler,
        "scenario_matcher": scenario_matcher,
        "route_decider": route_decider,
        "scenario_answer": scenario_answer,
        "rag3x_answer": rag3x_answer,
        "rag_result_evaluator": rag_result_evaluator,
        "web_search_answer": web_search_answer,
        "final_formatter": final_formatter,
    }

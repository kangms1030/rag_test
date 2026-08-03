"""LangGraph 메인그래프 노드 클로저 make_nodes(ctx).

각 노드는 ChatState 부분 업데이트(dict)를 반환한다. 비직렬화 객체(엔진/락)는
상태에 넣지 않고 ctx 를 통해 접근한다. 모든 라우팅 판단은 여기(+routing.py)에서 이뤄진다.

Phase 1: v1 동등(parity). rag3x_answer 는 blackbox 어댑터(ctx.rag_adapter) 호출.
Phase 2 에서 rag3x_answer 가 RAG 서브그래프 호출로 교체된다.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, interrupt

from ..scenario.matcher import normalize_text
from ..scenario.tree import InvalidActionError
from .routing import decide_route, evaluate_rag_result
from .state import ChatState, new_turn_defaults


def _format_history(messages: list, limit_chars: int = 2000) -> str:
    """최근 대화 이력을 '역할: 내용' 텍스트로(길이 상한). contextualize 프롬프트용."""
    rows = []
    for m in messages:
        role = "사용자" if isinstance(m, HumanMessage) else "상담봇"
        content = (getattr(m, "content", "") or "").strip()
        if content:
            rows.append(f"{role}: {content}")
    text = "\n".join(rows)
    return text[-limit_chars:] if len(text) > limit_chars else text


_PERSONA_PROMPT_NAMES = (
    "persona/persona",
    "persona/response_policy",
    "persona/response_examples",
)


def _render_persona_prompts(ctx: Any, route: str) -> tuple[str, list[str]]:
    """독립 페르소나 프롬프트를 합성 프롬프트 앞에 선택적으로 붙인다.

    파일이 없거나 ``PERSONA_PROMPTS_ENABLED=false``이면 빈 문자열을 반환한다.
    따라서 이 확장 파일을 수정·삭제해도 기존 composer 프롬프트와 LangGraph
    경로는 그대로 사용할 수 있다.
    """
    settings = getattr(ctx, "settings", None)
    if settings is not None and not getattr(settings, "persona_prompts_enabled", True):
        return "", []
    prompts = getattr(ctx, "prompts", None)
    if prompts is None:
        return "", []

    parts: list[str] = []
    loaded: list[str] = []
    for name in _PERSONA_PROMPT_NAMES:
        text = prompts.render_optional(name, route=route).strip()
        if text:
            parts.append(text)
            loaded.append(f"{name}.md")
    return "\n\n".join(parts), loaded

_CITE_RE = re.compile(r"\[\s*[pP]\s*(\d{1,4})\s*\]")


def _clean_citations(answer: str, evidence: list[dict]) -> tuple[str, list[int]]:
    """답변의 `[p53]` 출처 표기를 실제 근거 페이지와 대조해 정리한다 (2026-07-27 작업 9).

    모델이 근거에 없는 쪽번호를 지어내면 그 표기만 제거한다(문장은 남긴다 — 내용 자체는
    다른 근거에서 왔을 수 있다). 유효한 표기는 그대로 두고 목록을 반환해 프론트가
    근거 이미지 링크로 렌더링할 수 있게 한다.
    """
    valid = {int(e.get("page_number")) for e in (evidence or [])
             if str(e.get("page_number") or "").isdigit()}
    used: list[int] = []

    def _sub(m: re.Match) -> str:
        pg = int(m.group(1))
        if pg in valid:
            if pg not in used:
                used.append(pg)
            return f"[p{pg}]"
        return ""      # 근거에 없는 쪽번호 표기는 제거

    cleaned = _CITE_RE.sub(_sub, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,)])", r"\1", cleaned)
    return cleaned.strip(), used


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
        from ..observability.langsmith import add_node_metadata

        add_node_metadata(meta, tags=tags)
    except Exception:
        pass


def _trace(state: ChatState, node: str, detail: str) -> list:
    rows = list(state.get("trace") or [])
    rows.append({"node": node, "detail": detail})
    return rows


def _writer():
    """현재 노드의 custom stream writer(없으면 None). 스트리밍 중이 아니면 no-op."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001 - 스트리밍 컨텍스트 밖
        return None


def _progress(stage: str, msg: str) -> None:
    """SSE 진행상황 1건 발신(스트리밍이 아니면 조용히 무시)."""
    w = _writer()
    if w is None:
        return
    try:
        w({"stage": stage, "msg": msg})
    except Exception:  # noqa: BLE001
        pass


def make_nodes(ctx: Any) -> dict[str, Callable[[ChatState], dict]]:
    tree = ctx.tree
    faq = ctx.faq
    matcher = ctx.matcher
    settings = ctx.settings

    from ..rag.adapter_util import make_evidence_copier

    _evidence_copier = make_evidence_copier(settings.evidence_root)

    def _faq_run_id(faq_id: str) -> str:
        """faq_id('스쿨넷:2')는 URL 안전하지 않으므로 결정론적 해시 폴더명으로."""
        import hashlib

        return "faq" + hashlib.md5(faq_id.encode("utf-8")).hexdigest()[:12]

    def _faq_source_docs(faq_id: str) -> list[dict]:
        """FAQ 근거 문서 목록(코퍼스에서 식별된 것). 쪽번호 미인용이면 pages 는 빈 리스트.

        실측: FAQ 236행 중 199행이 쪽번호 없이 문서명만 인용한다 → 이미지는 못 붙여도
        '어떤 문서를 근거로 한 답변인지'는 보여줄 수 있다.
        """
        rows = (ctx.faq_links or {}).get(faq_id) or []
        return [
            {"document_name": r.get("document_name"),
             "pages": [p.get("page") for p in (r.get("pages") or [])]}
            for r in rows
        ]

    def _faq_evidence(faq_id: str, max_images: int = 3) -> list[dict]:
        """faq_doc_links 로 근거 페이지 이미지를 복사해 URL 목록 반환(LLM 0회, 파일 복사만).

        빌드 시점에 저장한 image_rel(파싱캐시 기준 상대경로)만 쓰므로 ragcore 로딩이 필요 없다.
        """
        rows = (ctx.faq_links or {}).get(faq_id) or []
        if not rows:
            return []
        run_id = _faq_run_id(faq_id)
        out: list[dict] = []
        rank = 1
        for row in rows:
            for pg in row.get("pages") or []:
                if rank > max_images:
                    return out
                rel = pg.get("image_rel")
                if not rel:
                    continue
                src = settings.parsed_dir / rel
                if not src.is_file():
                    continue
                urls = _evidence_copier(
                    {"_run_id": run_id, "page_number": pg.get("page"),
                     "page_image_resolved": str(src)},
                    rank,
                )
                if urls.get("image_url"):
                    out.append({
                        "rank": rank,
                        "document_name": row.get("document_name"),
                        "page_number": pg.get("page"),
                        "image_url": urls["image_url"],
                        "table_url": None,
                    })
                    rank += 1
        return out

    def _unsupported_claims(answer: str, context: str) -> list[str]:
        """합성문이 근거에 없는 숫자/코드를 새로 만들어냈는지(LLM 0회 결정론 대조).

        rag3.verify.check_claims_supported 재사용. 임포트 실패 시 검사 생략([]).
        """
        try:
            from ..rag.adapter_util import prepare_ragcore_imports

            prepare_ragcore_imports(settings)
            from rag3.verify import check_claims_supported

            return check_claims_supported(answer or "", context or "")
        except Exception:  # noqa: BLE001
            return []

    # ---------- 1. normalize_input ----------
    def normalize_input(state: ChatState) -> dict:
        out = new_turn_defaults()
        out["_turn_started_at"] = time.time()
        input_type = state.get("input_type") or "text"

        if input_type == "action":
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
            # 대화 메모리: 사용자 발화를 이력에 추가(add_messages 리듀서가 누적).
            out["messages"] = [HumanMessage(content=raw)]
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
            # (messages 는 절대 초기화하지 않는다 — 대화 메모리 유지.)
            out.update(
                scenario_id=None,
                current_node_id=None,
                scenario_completed=False,
                scenario_path=[],
            )
            detail = "자유 입력 → 시나리오 상태 중단"
        else:
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

    # ---------- 3.5 contextualize_query (Phase 3, 후속질문 재작성) ----------
    def contextualize_query(state: ChatState) -> dict:
        """자유입력 & 후속질문일 때만 대화 이력으로 자립형 질문 재작성.

        게이트: settings.contextualize_enabled ∧ text ∧ 이전 사용자 발화 ≥1(=후속).
        그 외에는 pass-through(standalone_question=원문). LLM 실패 시에도 원문 유지.
        """
        if state.get("input_type") != "text":
            return {}
        user_q = (state.get("user_input") or "").strip()
        msgs = state.get("messages") or []
        humans = [m for m in msgs if isinstance(m, HumanMessage)]
        if not settings.contextualize_enabled or len(humans) < 2:
            return {"standalone_question": user_q, "contextualized": False}

        _progress("contextualize", "이전 대화를 참고해 질문을 정리하고 있어요…")
        history = _format_history(msgs[:-1], limit_chars=2000)
        rewritten = user_q
        if ctx.llm is not None and ctx.prompts is not None:
            prompt = ctx.prompts.render("contextualize", history=history, question=user_q)
            out = ctx.llm.chat(prompt)
            if out:
                # 모델이 설명을 덧붙이는 경우 방어 — 첫 줄만 사용
                rewritten = out.splitlines()[0].strip().strip('"').strip() or user_q
        if rewritten and rewritten != user_q:
            _node_meta({"rewritten": True, "standalone_question": rewritten}, tags=["contextualized"])
            return {
                "standalone_question": rewritten,
                "normalized_question": normalize_text(rewritten),
                "contextualized": True,
                "trace": _trace(state, "contextualize_query", f"후속질문 재작성 → {rewritten[:40]}"),
            }
        return {
            "standalone_question": user_q,
            "contextualized": False,
            "trace": _trace(state, "contextualize_query", "재작성 불필요(원문 유지)"),
        }

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
        route, reason = decide_route(
            state,
            clarify_enabled=settings.clarify_enabled,
            clarify_min_score=settings.clarify_min_score,
        )
        _node_meta({"route": route, "route_reason": reason}, tags=[f"route:{route}"])
        return {
            "route": route,
            "route_reason": reason,
            "trace": _trace(state, "route_decider", f"route={route} ({reason})"),
        }

    # ---------- 5.5 clarify_node (Phase 3, HITL 인터럽트) ----------
    def clarify_node(state: ChatState) -> Command:
        """애매 매칭 시 상위 후보를 제시하고 실행을 일시정지(interrupt).

        재개(Command(resume={"choice": faq_id | "__none__"})) 시:
          - 후보 선택 → scenario_match 를 해당 FAQ accept 로 덮어쓰고 route="faq" → scenario_answer.
          - "__none__" → route="rag3x" → rag3x_answer.
        interrupt() 이전 코드는 재개 시 재실행되므로 부작용을 두지 않는다(후보 조회만).
        """
        norm = state.get("normalized_question") or ""
        candidates = matcher.top_candidates(norm, k=2)
        # 여기서 일시정지. resume 값이 choice 로 반환된다.
        resume = interrupt({"type": "clarify", "candidates": candidates})
        choice = (resume or {}).get("choice") if isinstance(resume, dict) else resume

        if choice and choice != "__none__" and faq.get(choice) is not None:
            entry = faq.get(choice)
            new_match = {
                "decision": "accept",
                "decision_reason": "clarify 되묻기에서 사용자가 후보 선택",
                "best_score": 1.0,
                "matched_id": entry.id,
                "matched_question": entry.question,
                "matched_sheet": entry.sheet,
                "matched_row": entry.row,
            }
            _node_meta({"clarify_choice": choice}, tags=["clarify_resolved:faq"])
            return Command(
                goto="scenario_answer",
                update={
                    "route": "faq",
                    "route_reason": "clarify 후 사용자 후보 선택",
                    "clarify_choice": choice,
                    "scenario_match": new_match,
                    "trace": _trace(state, "clarify_node", f"후보 선택 → {entry.id}"),
                },
            )
        _node_meta({"clarify_choice": "__none__"}, tags=["clarify_resolved:rag"])
        return Command(
            goto="rag3x_answer",
            update={
                "route": "rag3x",
                "route_reason": "clarify 후 '해당없음' → RAG",
                "clarify_choice": "__none__",
                "trace": _trace(state, "clarify_node", "해당없음 → RAG"),
            },
        )

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
            return {
                "final_answer": None,
                "answer_path": "none",
                "answer_source": "none",
                "confidence": "unknown",
                "trace": _trace(state, "scenario_answer", "FAQ 매칭 항목 없음(이상)"),
            }
        return {
            "final_answer": entry.answer,  # 원문 그대로(합성은 compose_answer 가 별도 필드로)
            "answer_path": "scenario",
            "answer_source": "faq_match",
            "confidence": "n/a",
            "options": [],
            "original_answer": entry.answer,
            "faq_evidence": _faq_evidence(entry.id),
            "source_meta": {
                "type": "faq",
                "sheet": entry.sheet,
                "row": entry.row,
                "no": entry.no,
                "question_type": entry.question_type,
                "fault_type": entry.fault_type,
                "source_files": entry.source_files,
                "evidence_docs": _faq_source_docs(entry.id),   # 코퍼스에서 식별된 근거 문서
                "matched_question": entry.question,
                "best_score": match.get("best_score"),
            },
            "trace": _trace(
                state, "scenario_answer",
                f"FAQ 모범답변 반환({entry.id}, score={match.get('best_score')})",
            ),
        }

    # ---------- 7. rag3x_answer (Phase 1: blackbox 어댑터) ----------
    def rag3x_answer(state: ChatState) -> dict:
        question = (state.get("standalone_question") or state.get("user_input") or "").strip()
        t0 = time.time()
        from ..observability.langsmith import traced_call

        _progress("rag", "내부 자료를 검색하고 있어요…")
        # RAG 서브그래프는 별도 Pregel 루프라 stream writer 가 전파되지 않는다.
        # 부모 노드에서 writer 를 잡아 콜백으로 넘겨 내부 단계도 스트리밍한다.
        writer = _writer()

        def _emit(ev: dict) -> None:
            if writer is not None:
                try:
                    writer(ev)
                except Exception:  # noqa: BLE001
                    pass

        def _do_ask(*, q):
            try:
                return ctx.rag_adapter.ask(q, progress=_emit)
            except TypeError:
                # progress 를 받지 않는 어댑터(구버전/테스트용)와의 호환
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
        question = (state.get("standalone_question") or state.get("user_input") or "").strip()
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

    # ---------- 9.5 compose_answer (Phase 4, 근거 종합 답변 작성) ----------
    def compose_answer(state: ChatState) -> dict:
        """FAQ 원문 / RAG 초안을 근거 기반으로 재구성해 composed_answer 에 담는다.

        - 시나리오 버튼 종단답변은 이 노드를 타지 않는다(결정론 절차 안내 원문 유지).
        - 합성 후 **결정론 숫자/코드 대조**(LLM 0회)로 근거 밖 수치가 생기면 합성을 폐기하고
          원문/초안을 그대로 쓴다(composer_fallback 에 사유 기록).
        """
        route = state.get("route")
        if route not in ("faq", "rag3x"):
            return {}
        enabled = (settings.composer_faq_enabled if route == "faq"
                   else settings.composer_rag_enabled)
        if not enabled or ctx.llm is None or ctx.prompts is None:
            return {"composed": False}

        question = (state.get("standalone_question") or state.get("user_input") or "").strip()
        history_summary = _format_history((state.get("messages") or [])[:-1], limit_chars=800)

        if route == "faq":
            base = (state.get("original_answer") or state.get("final_answer") or "").strip()
            if not base:
                return {"composed": False}
            evidence_text = base
            name = "composer_faq"
            prompt = ctx.prompts.render(
                name, question=question, original_answer=base, history_summary=history_summary
            )
        else:
            rag = state.get("rag_result") or {}
            base = (rag.get("final_answer") or "").strip()
            evidence_text = rag.get("answer_context") or ""
            if not base:
                return {"composed": False}
            name = "composer_rag"
            prompt = ctx.prompts.render(
                name, question=question, answer=base,
                evidence_text=evidence_text, history_summary=history_summary,
            )

        # 추가 페르소나/응답정책은 기존 composer 파일과 분리된 선택적 확장이다.
        # 파일이 없거나 환경변수로 끄면 위에서 만든 기존 prompt를 그대로 사용한다.
        persona_text, persona_files = _render_persona_prompts(ctx, route)
        if persona_text:
            prompt = f"{persona_text}\n\n{prompt}"

        _progress("compose", "찾은 근거를 종합해 답변을 정리하고 있어요…")
        t0 = time.time()
        out = ctx.llm.chat(prompt)
        timings = dict(state.get("timings") or {})
        timings["compose_s"] = time.time() - t0

        if not out:
            return {"composed": False, "composer_fallback": "LLM 미응답 → 원문 유지",
                    "timings": timings,
                    "trace": _trace(state, "compose_answer", "합성 실패(LLM 미응답) → 원문 유지")}

        # 검증 컨텍스트 = 근거 + 초안 + **프롬프트 원본 템플릿**.
        # 템플릿에는 우리가 고정으로 넣은 안내 상수(예: 지원센터 1899-0979)가 들어 있다.
        # 이를 빼면 모델이 지시를 따라 그 번호를 인용했을 때 '근거 밖 수치'로 오탐해
        # 정상 합성을 폐기한다(실측으로 확인). 템플릿은 치환 전이라 사용자 입력은 포함되지 않는다.
        try:
            template_text = ctx.prompts.load(name).template
        except Exception:  # noqa: BLE001
            template_text = ""
        unsupported = _unsupported_claims(out, f"{evidence_text}\n{base}\n{template_text}")
        meta = {"composer": name, "route": route, "unsupported": unsupported[:5]}
        meta["persona_prompts_enabled"] = bool(persona_files)
        meta["persona_prompt_files"] = persona_files
        meta.update(ctx.prompts.meta(name) if ctx.prompts else {})
        if unsupported:
            _node_meta(meta, tags=["composed:fallback"])
            return {
                "composed": False,
                "composer_fallback": f"근거 밖 수치 {unsupported[:5]} 발견 → 합성 폐기",
                "timings": timings,
                "trace": _trace(state, "compose_answer",
                                f"합성 폐기(근거 밖 수치 {unsupported[:3]}) → 원문 유지"),
            }
        # 2026-07-27(작업 9): RAG 경로는 문장별 출처 표기 [p53] 을 실제 근거와 대조해 정리한다.
        citations: list[int] = []
        if route == "rag3x":
            rag = state.get("rag_result") or {}
            out, citations = _clean_citations(out, rag.get("evidence") or [])
            meta["citations"] = citations
        _node_meta(meta, tags=["composed:ok"])
        return {
            "composed": True,
            "composed_answer": out,
            "citations": citations,
            "timings": timings,
            "trace": _trace(state, "compose_answer",
                            f"{name} 합성 채택(len={len(out)}, 출처 {len(citations)}쪽)"),
        }

    # ---------- 9.6 answer_grader (Phase 4, 해결도 판정 + 에스컬레이션) ----------
    def answer_grader(state: ChatState) -> dict:
        """답변이 질문을 해결했는지 판정.

        - FAQ 경로: 미해결이면 RAG 로 1회 에스컬레이션(기존 동작).
        - RAG 경로(2026-07-27 신설): 판정만 하고 **재시도는 하지 않는다**(지연·루프 방지).
          미해결이면 경고 문구를 덧붙이고 confidence 를 낮춘다.

        왜 RAG 에도 붙였나 — 원래 주석은 "rag3x 내부 verify 가 이미 신뢰도를 판정한다"였으나,
        그 verify 는 abstain 오탐 때문에 무력화돼 있었다(작업 1). 결과적으로 RAG 답변에는
        어떤 의미 수준 검증도 걸리지 않았다. answer_grader.md 의 UNRESOLVED 정의
        ("주제는 비슷하나 요구한 핵심 정보가 빠졌거나")가 이번 실패 유형과 정확히 일치한다.
        """
        route = state.get("route")
        # 건너뛸 때는 grader_verdict 를 덮어쓰지 않는다 — 에스컬레이션으로 RAG 재시도 중이면
        # 1차(FAQ) 판정 기록이 최종 상태에 남아야 관측 가능하다.
        if (not settings.grader_enabled or route not in ("faq", "rag3x")
                or ctx.llm is None or ctx.prompts is None):
            return {"_escalate": False}

        answer = (state.get("composed_answer") or state.get("final_answer") or "").strip()
        question = (state.get("standalone_question") or state.get("user_input") or "").strip()
        if not answer:
            return {"_escalate": False}

        _progress("grade", "답변이 질문을 해결했는지 확인하고 있어요…")
        raw = (ctx.llm.chat(ctx.prompts.render("answer_grader", question=question, answer=answer))
               or "").upper()
        # 주의: "UNRESOLVED" 안에 "RESOLVED" 가 포함되므로 반드시 UNRESOLVED 를 먼저 검사.
        if "UNRESOLVED" in raw:
            verdict = "unresolved"
        elif "RESOLVED" in raw:
            verdict = "resolved"
        else:
            verdict = None

        # RAG 경로는 에스컬레이션하지 않는다 — 같은 질문으로 RAG 를 다시 돌리면 결과가 같고
        # 지연만 2배가 된다(캐시가 있으면 완전히 동일). 판정 결과만 사용자에게 surface 한다.
        if route == "rag3x":
            _node_meta({"grader_verdict": verdict, "escalate": False},
                       tags=[f"grader:{verdict or 'unknown'}"])
            out: dict = {"grader_verdict": verdict, "_escalate": False,
                         "trace": _trace(state, "answer_grader", f"RAG 판정={verdict or 'unknown'}")}
            if verdict == "unresolved":
                out["confidence"] = "low"
                out["warnings"] = list(state.get("warnings") or []) + [
                    "이 답변이 질문의 핵심을 충분히 다루지 못했을 수 있습니다. "
                    "질문을 더 구체적으로 다시 물어보시거나, 담당 선생님 또는 "
                    "스쿨넷 서비스 지원센터(1899-0979)로 문의해 주세요."
                ]
            return out

        escalate = verdict == "unresolved" and int(state.get("escalate_budget") or 0) > 0
        _node_meta({"grader_verdict": verdict, "escalate": escalate},
                   tags=[f"grader:{verdict or 'unknown'}"])
        if escalate:
            return {
                "grader_verdict": verdict,
                "_escalate": True,
                "escalate_budget": 0,          # 1회로 제한(무한루프 방지)
                "composed_answer": None,       # RAG 경로에서 새로 합성
                "composed": False,
                "trace": _trace(state, "answer_grader", "FAQ 답변 미해결 → RAG 재시도(1회)"),
            }
        return {
            "grader_verdict": verdict,
            "_escalate": False,
            "trace": _trace(state, "answer_grader", f"판정={verdict or 'unknown'}"),
        }

    # ---------- 10. final_formatter ----------
    def final_formatter(state: ChatState) -> dict:
        route = state.get("route")
        out: dict = {}

        composed_answer = state.get("composed_answer")

        if route in ("scenario", "faq"):
            # scenario_answer 가 이미 final_answer/source_meta 를 채움 — 그대로 유지.
            # 단 FAQ 합성이 채택됐으면 표시 답변만 교체(원문은 original_answer 로 동봉).
            out["evidence"] = []
            out["verification"] = None
            if route == "faq" and composed_answer:
                out["final_answer"] = composed_answer
        elif route == "rag3x":
            rag = state.get("rag_result") or {}
            # 2026-07-27: answer_grader(RAG 경로)가 unresolved 로 판정하면 그 결과가 우선한다.
            # rag3x 내부 confidence 는 근거 정합성(숫자대조·groundedness)만 보고, 질문을
            # 실제로 해결했는지는 보지 않기 때문이다.
            conf = rag.get("confidence") or "unknown"
            if state.get("grader_verdict") == "unresolved":
                conf = "low"
            out.update(
                final_answer=composed_answer or rag.get("final_answer"),
                answer_path=rag.get("answer_path") or "none",
                answer_source="rag3x",
                confidence=conf,
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

        timings = dict(state.get("timings") or {})
        started = state.get("_turn_started_at") or time.time()
        total = time.time() - started
        timings["total_s"] = total
        out["timings"] = timings

        # 대화 메모리: 상담봇 최종 답변을 이력에 추가(빈 답변/시나리오 메뉴 텍스트도 포함).
        ans_text = out.get("final_answer")
        if ans_text is None:
            ans_text = state.get("final_answer")
        if ans_text:
            out["messages"] = [AIMessage(content=str(ans_text))]

        out["trace"] = _trace(state, "final_formatter", f"route={route}, total={total:.3f}s")

        try:
            from ..observability.langsmith import attach_run_metadata

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
        "contextualize_query": contextualize_query,
        "scenario_matcher": scenario_matcher,
        "route_decider": route_decider,
        "clarify_node": clarify_node,
        "scenario_answer": scenario_answer,
        "rag3x_answer": rag3x_answer,
        "rag_result_evaluator": rag_result_evaluator,
        "web_search_answer": web_search_answer,
        "compose_answer": compose_answer,
        "answer_grader": answer_grader,
        "final_formatter": final_formatter,
    }

"""RAG 서브그래프 노드 (controller_x S0~S8 을 LangGraph 노드로 편성).

각 노드는 vendored rag3/rag3x 원시함수만 호출한다(무수정 재사용). controller.py 의
answer_question 상태기계를 노드/엣지로 펼쳐, LangSmith 에서 retrieve→(crag)→answer→verify→
(rollback)→finalize 가 각각 child run 으로 보이게 한다.

무한루프 방지:
- CRAG 재검색은 crag_budget(=1)로 1회 제한(retrieve 로 되돌아가는 유일한 사이클).
- 롤백은 controller 와 동일하게 단발(전용 노드 A/B/C 각 1회, 이후 finalize). 루프 없음.
- 모델 호출 상한(_model_calls<5/4)·deadline 도 controller 그대로 검사.

metrics: 각 노드 본문은 build_rag_subgraph 에서 with metrics.run_metrics(m) 로 감싸므로
(m=deps.scratch['metrics']), rag3 원시함수의 record_* 가 정상 누산된다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .state import RagState


@dataclass
class RagDeps:
    """RAG 서브그래프 노드가 쓰는 런타임 의존성(비직렬화 — 상태 대신 여기로)."""
    config: Any
    backend: Any
    scratch: dict = field(default_factory=dict)   # 현재 run 의 metrics(RunMetrics) 등


def _retrieval_to_dict(rr) -> dict:
    return {
        "selected_pages": rr.selected_pages,
        "selected_documents": rr.selected_documents,
        "answer_path": rr.answer_path,
        "route_reason": rr.route_reason,
        "rerank_top_score": rr.rerank_top_score,
    }


def make_rag_nodes(deps: RagDeps) -> tuple[dict[str, Callable], dict[str, Callable]]:
    """노드 dict 와 라우팅함수 dict 를 반환. (sys.path 에 ragcore 가 있어야 import 성공)"""
    from rag3 import metrics
    from rag3.answer import answer_vision_from_page
    from rag3.controller import _NO_ANSWER, _finalize, _model_calls, _verify
    from rag3.evidence import resolve_evidence
    from rag3.judge import rewrite_query
    from rag3.retrieve import RetrievalResult, run_retrieval
    from rag3.verify import is_abstain
    from rag3x.controller_x import _decide_ground

    config = deps.config
    backend = deps.backend

    def _emit(stage: str, msg: str) -> None:
        """SSE 진행상황 발신. 부모 노드가 scratch['progress'] 로 콜백을 넣어줄 때만 동작."""
        cb = deps.scratch.get("progress")
        if cb is None:
            return
        try:
            cb({"stage": stage, "msg": msg})
        except Exception:  # noqa: BLE001
            pass

    # ---------- prepare ----------
    def prepare(state: RagState) -> dict:
        t0 = time.time()
        return {
            "started_ts": t0,
            "deadline_ts": t0 + config.deadline_seconds,
            "crag_budget": 1,
            "answer_regen_budget": 1,
            "path_switch_budget": 1,
            "history": [],
        }

    # ---------- retrieve (S1-S5) ----------
    def retrieve(state: RagState) -> dict:
        q = state.get("rewritten_query") or state["question"]
        is_first = not state.get("rewritten_query")
        _emit("retrieve", "근거 문서를 검색하고 리랭킹하는 중…" if is_first
              else "재작성한 질문으로 다시 검색하는 중…")
        tr = time.time()
        rr = run_retrieval(q, config, backend)
        if is_first:
            metrics.record_timing("retrieve", time.time() - tr)
        return {"retrieval": _retrieval_to_dict(rr)}

    # ---------- crag_rewrite (S3a, 사이클) ----------
    def crag_rewrite(state: RagState) -> dict:
        _emit("crag", "근거를 못 찾아 질문을 다시 쓰는 중…")
        rq = rewrite_query(state["question"], backend, config)
        rr = state["retrieval"]
        return {
            "rewritten_query": rq,
            "crag_budget": 0,
            "history": [{
                "action": "crag_rewrite", "rewritten_query": rq,
                "old_top": rr.get("rerank_top_score"),
            }],
        }

    # ---------- answer (S6) ----------
    def answer_node(state: RagState) -> dict:
        rr = state["retrieval"]
        pages = rr["selected_pages"]
        path = rr["answer_path"]
        _emit("answer", f"근거 {len(pages)}페이지로 답변을 생성하는 중…"
              if path == "text" else "표·도표 이미지를 읽어 답변을 생성하는 중…")
        ta = time.time()
        if path == "vision":
            ans = answer_vision_from_page(state["question"], pages[0], backend, config)
        else:
            from rag3x.answer_x import answer_text_from_pages_x
            ans = answer_text_from_pages_x(state["question"], pages, backend, config)
        metrics.record_timing("answer", time.time() - ta)
        return {"answer": ans, "answer_path": path}

    # ---------- verify (S7) ----------
    def verify_node(state: RagState) -> dict:
        rr = state["retrieval"]
        pages = rr["selected_pages"]
        path = state["answer_path"]
        top = rr.get("rerank_top_score") or 0.0
        _emit("verify", "답변이 근거와 맞는지 검증하는 중…")
        ground = _decide_ground(state["answer"], path, top, rr["selected_documents"], config)
        v = _verify(state["answer"], path, pages[0], backend, config, ground=ground)
        return {"verify": v}

    # ---------- rollback A: text 빈응답/확인불가 → 1순위 페이지 단독 재시도 ----------
    def rollback_top1(state: RagState) -> dict:
        _emit("rollback", "답변이 부실해 1순위 근거로 다시 시도하는 중…")
        from rag3x.answer_x import answer_text_from_pages_x
        rr = state["retrieval"]
        pages = rr["selected_pages"]
        m = metrics.current()
        ans2 = answer_text_from_pages_x(state["question"], pages[:1], backend, config)
        out: dict = {"answer_regen_budget": 0,
                     "history": [{"action": "rollback_text_top1", "trigger": "abstain"}]}
        if not is_abstain(ans2["final_answer"]):
            out["answer"] = ans2
            out["answer_path"] = "text"
            out["verify"] = _verify(ans2, "text", pages[0], backend, config,
                                    ground=(_model_calls(m) < 4))
        return out

    # ---------- rollback B: text 숫자 미지원 + 스캔 → vision 교차확인 ----------
    def rollback_vision(state: RagState) -> dict:
        _emit("rollback", "숫자 검증에 걸려 원본 이미지로 교차확인하는 중…")
        rr = state["retrieval"]
        pages = rr["selected_pages"]
        ansv = answer_vision_from_page(state["question"], pages[0], backend, config)
        verv = _verify(ansv, "vision", pages[0], backend, config, ground=False)
        out: dict = {"path_switch_budget": 0,
                     "history": [{"action": "rollback_text_to_vision", "trigger": "unsupported_numbers"}]}
        if verv is not None and not verv["transcription_ocr_mismatch"] and not verv["abstain"]:
            out["answer"] = ansv
            out["answer_path"] = "vision"
            out["verify"] = verv
        return out

    # ---------- rollback C: vision 전사-OCR 불일치 → OCR 텍스트 재구성 ----------
    def rollback_ocr(state: RagState) -> dict:
        _emit("rollback", "이미지 전사가 OCR과 달라 텍스트로 재구성하는 중…")
        from rag3x.answer_x import answer_text_from_pages_x
        rr = state["retrieval"]
        pages = rr["selected_pages"]
        m = metrics.current()
        ans2 = answer_text_from_pages_x(state["question"], pages[:1], backend, config)
        out: dict = {"path_switch_budget": 0,
                     "history": [{"action": "rollback_vision_to_ocr", "trigger": "transcription_ocr_mismatch"}]}
        if not is_abstain(ans2["final_answer"]):
            out["answer"] = ans2
            out["answer_path"] = "text"
            out["verify"] = _verify(ans2, "text", pages[0], backend, config,
                                    ground=(_model_calls(m) < 4))
        return out

    # ---------- finalize (S8 종료) ----------
    def finalize(state: RagState) -> dict:
        rr = state.get("retrieval") or {"selected_pages": [], "selected_documents": [],
                                        "route_reason": "", "rerank_top_score": None}
        retr = RetrievalResult(
            question=state["question"],
            selected_documents=rr["selected_documents"],
            selected_pages=rr["selected_pages"],
            answer_path=rr["answer_path"] if "answer_path" in rr else "none",
            route_reason=rr.get("route_reason", ""),
            rerank_top_score=rr.get("rerank_top_score"),
        )
        history = list(state.get("history") or [])
        m = metrics.current()
        if history and m is not None:
            m.rollback_count = len([h for h in history if h.get("action") != "crag_rewrite"])

        if state.get("answer") is None:
            final_answer, path, verify = _NO_ANSWER, "none", None
        else:
            final_answer = state["answer"]["final_answer"]
            path = state.get("answer_path") or "text"
            verify = state.get("verify")

        started = state.get("started_ts") or time.time()
        result = _finalize(state["question"], retr, final_answer, path, verify,
                           history, m, state.get("run_id"), config, started)
        # composer(Phase 4)가 근거로 쓸 컨텍스트. 페이지 텍스트만 담기며 파일경로는 없다.
        ans_ctx = (state.get("answer") or {}).get("context") or ""
        result["answer_context"] = ans_ctx[: int(getattr(config, "context_max_chars", 10000))]
        # Gemini 토큰/비용 surface (controller_x 와 동일 — 로컬 백엔드면 no-op)
        acc = getattr(m, "_gemini", None) if m is not None else None
        if acc:
            result["metrics"]["gemini_tokens_in"] = acc.get("in")
            result["metrics"]["gemini_tokens_out"] = acc.get("out")
            result["metrics"]["gemini_calls"] = acc.get("calls")
            result["metrics"]["gemini_cost"] = round(acc.get("cost", 0.0), 6)
            result["metrics"]["gemini_api_s"] = round(acc.get("api_s", 0.0), 2)
        resolve_evidence(result, config)
        return {"result": result}

    # ---------- 라우팅 함수 ----------
    def after_retrieve(state: RagState) -> str:
        rr = state["retrieval"]
        if rr["answer_path"] != "none":
            return "answer"
        top = rr.get("rerank_top_score") or 0.0
        if (getattr(config, "enable_crag", False) and state.get("crag_budget", 0) > 0
                and config.crag_retry_floor <= top < config.rerank_score_floor):
            return "crag"
        return "no_answer"

    def after_verify(state: RagState) -> str:
        m = metrics.current()
        verify = state.get("verify")
        if (not getattr(config, "enable_rollback", False) or time.time() >= state.get("deadline_ts", 0)
                or (m is not None and _model_calls(m) >= 5) or verify is None):
            return "done"
        rr = state["retrieval"]
        pages = rr["selected_pages"]
        path = state["answer_path"]
        top = rr.get("rerank_top_score") or 0.0
        det_unsupported = bool(verify["unsupported_claims"])
        det_abstain = bool(verify["abstain"])
        det_ocr = bool(verify["transcription_ocr_mismatch"])
        mc = _model_calls(m) if m is not None else 0

        if (path == "text" and det_abstain and top >= config.rollback_rerank_tau_high
                and state.get("answer_regen_budget", 0) > 0 and len(pages) >= 1):
            return "rollback_top1"
        if (path == "text" and det_unsupported and pages[0].get("is_scanned")
                and state.get("path_switch_budget", 0) > 0 and mc < 4):
            return "rollback_vision"
        if (path == "vision" and det_ocr and pages[0].get("text")
                and state.get("path_switch_budget", 0) > 0 and mc < 4):
            return "rollback_ocr"
        return "done"

    nodes = {
        "prepare": prepare,
        "retrieve": retrieve,
        "crag_rewrite": crag_rewrite,
        "answer_node": answer_node,
        "verify_node": verify_node,
        "rollback_top1": rollback_top1,
        "rollback_vision": rollback_vision,
        "rollback_ocr": rollback_ocr,
        "finalize": finalize,
    }
    routers = {"after_retrieve": after_retrieve, "after_verify": after_verify}
    return nodes, routers

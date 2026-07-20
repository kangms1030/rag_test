"""컨트롤러 포크 (Phase 6) — S1~S8 상태기계에 실험 훅을 얹는다.

원본: rag3/controller.py (answer_question). 헬퍼(_model_calls/_verify/_finalize/_NO_ANSWER)와
검색·CRAG·롤백 구조는 원본을 그대로 재사용/미러링한다. 추가된 실험 훅은 전부 플래그 게이트이며
**전 플래그 OFF면 원본 answer_question과 동일 동작**(등가성 계약, Phase 6.0에서 실측 증명).

실험 훅:
- P1(a) x_fail_fast_on_length_budget: length-retry 예산 소진(=병리적 표문맥) 문항은 S8 롤백을
  생략하고 즉시 확정한다(FINAL §5-B — 해당 문항은 파싱천장이라 abstain이 정답).
- P1(b) x_conditional_verify_skip: 숫자대조 통과 ∧ rerank 고점 ∧ 단일문서면 S7 groundedness
  LLM 호출을 생략(결정론 숫자대조는 그대로 수행 → 환각 0 방어는 유지).
- P1(c) 적응형 트림: answer_x.answer_text_from_pages_x에서 처리.
- (P3) 분해 라우팅/문장인용은 후속 커밋에서 이 파일에 추가.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from rag3 import metrics
from rag3.answer import answer_vision_from_page
from rag3.config import Config
from rag3.controller import _NO_ANSWER, _finalize, _model_calls, _verify
from rag3.judge import rewrite_query
from rag3.models import Backend
from rag3.retrieve import RetrievalResult, run_retrieval
from rag3.verify import check_claims_supported, is_abstain

from .answer_x import answer_text_from_pages_x
from .decompose import decompose_answer, should_decompose

logger = logging.getLogger(__name__)


def _decide_ground(ans: dict, path: str, top_score: float, selected_documents: list, config: Config) -> bool:
    """P1(b): S7 groundedness LLM 호출 여부. 플래그 OFF면 항상 True(원본과 동일)."""
    if not getattr(config, "x_conditional_verify_skip", False):
        return True
    if path != "text":
        return True
    single_doc = len(selected_documents) <= 1
    tau = float(getattr(config, "x_verify_skip_tau", 0.6))
    ans_text = ans.get("final_answer", "")
    if (single_doc and top_score >= tau
            and not is_abstain(ans_text)
            and not check_claims_supported(ans_text, ans.get("context", ""))):
        return False  # 숫자대조 통과 + 고점 + 단일문서 → groundedness 스킵
    return True


def answer_question(question: str, config: Config, backend: Backend, *, run_id: str | None = None) -> dict[str, Any]:
    m = metrics.RunMetrics()
    with metrics.run_metrics(m):
        t0 = time.time()
        deadline = t0 + config.deadline_seconds
        history: list[dict[str, Any]] = []
        budget = {"crag": 0, "answer_regen": 0, "path_switch": 0}

        # --- S0 라우팅: 종합형(복합) 질문만 분해검색 경로로. 단일형은 기존 경로 그대로(무회귀) ---
        if should_decompose(question, config):
            ta = time.time()
            d = decompose_answer(question, config, backend)
            metrics.record_timing("answer", time.time() - ta)
            pages = d["selected_pages"]
            if not pages:
                rr = RetrievalResult(question, [], [], answer_path="none", route_reason="decompose: 근거없음")
                return _finalize(question, rr, _NO_ANSWER, "none", None,
                                 [{"action": "decompose", "subquestions": d.get("subquestions")}],
                                 m, run_id, config, t0)
            seen: dict[str, dict] = {}
            for sp in pages:
                dn = sp["document_name"]
                if dn not in seen:
                    seen[dn] = {"rank": len(seen) + 1, "document_name": dn, "selection_score": sp.get("page_score")}
            rr = RetrievalResult(question, list(seen.values()), pages, answer_path="text",
                                 route_reason=f"decompose: 하위질문 {len(d.get('subquestions', []))}개",
                                 rerank_top_score=(pages[0].get("page_score") if pages else None))
            verify = _verify(d, "text", pages[0], backend, config, ground=True)
            hist = [{"action": "decompose", "subquestions": d.get("subquestions"),
                     "dropped_sentences": d.get("dropped_sentences", [])}]
            result = _finalize(question, rr, d["final_answer"], "text", verify, hist, m, run_id, config, t0)
            acc = getattr(m, "_gemini", None)
            if acc:
                result["metrics"]["gemini_calls"] = acc["calls"]
                result["metrics"]["gemini_cost"] = round(acc["cost"], 6)
                result["metrics"]["gemini_api_s"] = round(acc.get("api_s", 0.0), 2)
            return result

        # --- S1-S5 검색 ---
        tr = time.time()
        retrieval = run_retrieval(question, config, backend)
        metrics.record_timing("retrieve", time.time() - tr)

        # --- CRAG: none + 경계 점수면 재작성 후 1회 재검색 ---
        if retrieval.answer_path == "none" and config.enable_crag and budget["crag"] == 0:
            top = retrieval.rerank_top_score or 0.0
            if config.crag_retry_floor <= top < config.rerank_score_floor:
                budget["crag"] += 1
                rq = rewrite_query(question, backend, config)
                r2 = run_retrieval(rq, config, backend)
                history.append({"action": "crag_rewrite", "rewritten_query": rq,
                                "old_top": top, "new_top": r2.rerank_top_score, "new_path": r2.answer_path})
                if r2.answer_path != "none":
                    retrieval = r2

        if retrieval.answer_path == "none":
            return _finalize(question, retrieval, _NO_ANSWER, "none", None, history, m, run_id, config, t0)

        pages = retrieval.selected_pages
        path = retrieval.answer_path
        top_score = retrieval.rerank_top_score or 0.0

        # --- S6 답변 (P1c 적응형 트림은 answer_text_from_pages_x 내부) ---
        ta = time.time()
        if path == "vision":
            ans = answer_vision_from_page(question, pages[0], backend, config)
        else:
            ans = answer_text_from_pages_x(question, pages, backend, config)
        metrics.record_timing("answer", time.time() - ta)

        # --- S7 검증 (P1b: 조건부 groundedness 스킵) ---
        ground = _decide_ground(ans, path, top_score, retrieval.selected_documents, config)
        verify = _verify(ans, path, pages[0], backend, config, ground=ground)

        # --- P1a: length-retry 예산 소진 문항은 S8 롤백 생략(즉시 확정) ---
        fail_fast = (getattr(config, "x_fail_fast_on_length_budget", False)
                     and m.length_retry_count >= getattr(config, "ollama_max_length_retries", 2))

        # --- S8 결정론 롤백(최대 1회, 트리거별 예산) ---
        if (config.enable_rollback and time.time() < deadline and _model_calls(m) < 5
                and verify is not None and not fail_fast):
            det_unsupported = bool(verify["unsupported_claims"])
            det_abstain = bool(verify["abstain"])
            det_ocr_mismatch = bool(verify["transcription_ocr_mismatch"])

            # A) text 빈응답/확인불가 + 검색 신뢰 -> 1순위 페이지 단독 재시도
            if (path == "text" and det_abstain and top_score >= config.rollback_rerank_tau_high
                    and budget["answer_regen"] == 0 and len(pages) >= 1):
                budget["answer_regen"] += 1
                ans2 = answer_text_from_pages_x(question, pages[:1], backend, config)
                history.append({"action": "rollback_text_top1", "trigger": "abstain"})
                if not is_abstain(ans2["final_answer"]):
                    ans = ans2
                    verify = _verify(ans, "text", pages[0], backend, config, ground=(_model_calls(m) < 4))

            # B) text 숫자 미지원 + 스캔 페이지 -> vision 교차확인 전환
            elif (path == "text" and det_unsupported and pages[0].get("is_scanned")
                    and budget["path_switch"] == 0 and _model_calls(m) < 4):
                budget["path_switch"] += 1
                ansv = answer_vision_from_page(question, pages[0], backend, config)
                verv = _verify(ansv, "vision", pages[0], backend, config, ground=False)
                history.append({"action": "rollback_text_to_vision", "trigger": "unsupported_numbers"})
                if verv is not None and not verv["transcription_ocr_mismatch"] and not verv["abstain"]:
                    ans, verify, path = ansv, verv, "vision"

            # C) vision 전사-OCR 불일치 -> OCR 텍스트로 재구성(1순위 페이지 text 답변)
            elif (path == "vision" and det_ocr_mismatch and pages[0].get("text")
                    and budget["path_switch"] == 0 and _model_calls(m) < 4):
                budget["path_switch"] += 1
                ans2 = answer_text_from_pages_x(question, pages[:1], backend, config)
                history.append({"action": "rollback_vision_to_ocr", "trigger": "transcription_ocr_mismatch"})
                if not is_abstain(ans2["final_answer"]):
                    ans, path = ans2, "text"
                    verify = _verify(ans, "text", pages[0], backend, config, ground=(_model_calls(m) < 4))

        if fail_fast:
            history.append({"action": "fail_fast_length_budget", "length_retries": m.length_retry_count})
        if history:
            m.rollback_count = len([h for h in history if h.get("action") != "fail_fast_length_budget"])
        result = _finalize(question, retrieval, ans["final_answer"], path, verify, history, m, run_id, config, t0)
        # Gemini 토큰/비용 surface (로컬 백엔드면 accumulator 없음 → no-op)
        acc = getattr(m, "_gemini", None)
        if acc:
            result["metrics"]["gemini_tokens_in"] = acc["in"]
            result["metrics"]["gemini_tokens_out"] = acc["out"]
            result["metrics"]["gemini_calls"] = acc["calls"]
            result["metrics"]["gemini_cost"] = round(acc["cost"], 6)
            result["metrics"]["gemini_api_s"] = round(acc.get("api_s", 0.0), 2)  # raw(스로틀 제외)
        return result

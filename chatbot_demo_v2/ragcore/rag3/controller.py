"""Phase 2 오케스트레이터: 검색 -> 답변 -> 검증 -> (결정론) 롤백. 상태기계 + 재시도 예산 + 데드라인.

계획 §무한루프 방지:
1. 각 재시도 0->1만 (crag/answer_regen/path_switch).
2. 총 모델 호출(임베딩 제외) 상한 5회 초과 시 즉시 현재 최선 답 반환.
3. 롤백 트리거는 결정론 신호만(빈응답/확인불가, 숫자 미지원, 전사-OCR 불일치). groundedness(LLM 판정)
   단독으로는 롤백하지 않고 신뢰도 태그만 부여.
4. wall-clock 데드라인(config.deadline_seconds) 초과 시 best-effort 반환.
모든 재시도는 rollback_history에 기록.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from . import metrics
from .answer import answer_text_from_pages, answer_vision_from_page
from .config import Config
from .models import Backend
from .judge import rewrite_query
from .retrieve import RetrievalResult, run_retrieval
from .verify import is_abstain, verify_answer

logger = logging.getLogger(__name__)

_NO_ANSWER = "선택된 문서에서 확인 불가"


def _model_calls(m: metrics.RunMetrics) -> int:
    return m.text_answer_calls + m.vision_answer_calls + m.judge_calls + m.verify_calls


def _verify(ans: dict, path: str, page0: dict | None, backend: Backend, config: Config, *, ground: bool) -> dict | None:
    if not config.enable_verify:
        return None
    return verify_answer(
        ans["final_answer"], ans.get("context", ""),
        answer_path=path, transcription=ans.get("transcription"),
        ocr_text=(page0.get("text") if (path == "vision" and page0) else None),
        backend=backend, config=config, run_groundedness=ground,
    )


def answer_question(question: str, config: Config, backend: Backend, *, run_id: str | None = None) -> dict[str, Any]:
    m = metrics.RunMetrics()
    with metrics.run_metrics(m):
        t0 = time.time()
        deadline = t0 + config.deadline_seconds
        history: list[dict[str, Any]] = []
        budget = {"crag": 0, "answer_regen": 0, "path_switch": 0}

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

        # --- S6 답변 ---
        ta = time.time()
        if path == "vision":
            ans = answer_vision_from_page(question, pages[0], backend, config)
        else:
            ans = answer_text_from_pages(question, pages, backend, config)
        metrics.record_timing("answer", time.time() - ta)

        # --- S7 검증 ---
        verify = _verify(ans, path, pages[0], backend, config, ground=True)

        # --- S8 결정론 롤백(최대 1회, 트리거별 예산) ---
        if config.enable_rollback and time.time() < deadline and _model_calls(m) < 5 and verify is not None:
            det_unsupported = bool(verify["unsupported_claims"])
            det_abstain = bool(verify["abstain"])
            det_ocr_mismatch = bool(verify["transcription_ocr_mismatch"])

            # A) text 빈응답/확인불가 + 검색 신뢰 -> 1순위 페이지 단독 재시도
            if (path == "text" and det_abstain and top_score >= config.rollback_rerank_tau_high
                    and budget["answer_regen"] == 0 and len(pages) >= 1):
                budget["answer_regen"] += 1
                ans2 = answer_text_from_pages(question, pages[:1], backend, config)
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
                # vision이 OCR과 모순되지 않고 회피하지 않을 때만 채택
                if verv is not None and not verv["transcription_ocr_mismatch"] and not verv["abstain"]:
                    ans, verify, path = ansv, verv, "vision"

            # C) vision 전사-OCR 불일치 -> OCR 텍스트로 재구성(1순위 페이지 text 답변)
            elif (path == "vision" and det_ocr_mismatch and pages[0].get("text")
                    and budget["path_switch"] == 0 and _model_calls(m) < 4):
                budget["path_switch"] += 1
                ans2 = answer_text_from_pages(question, pages[:1], backend, config)
                history.append({"action": "rollback_vision_to_ocr", "trigger": "transcription_ocr_mismatch"})
                if not is_abstain(ans2["final_answer"]):
                    ans, path = ans2, "text"
                    verify = _verify(ans, "text", pages[0], backend, config, ground=(_model_calls(m) < 4))

        if history:
            m.rollback_count = len(history)
        return _finalize(question, retrieval, ans["final_answer"], path, verify, history, m, run_id, config, t0)


def _finalize(question, retrieval: RetrievalResult, final_answer: str, path: str,
              verify: dict | None, history: list, m: metrics.RunMetrics, run_id, config: Config, t0: float) -> dict[str, Any]:
    metrics.record_timing("total", time.time() - t0)
    evidence = [
        {"document_name": p["document_name"], "page_number": p["page_number"],
         "page_image_path": p.get("page_image_path", ""), "table_crop_path": p.get("table_crop_path", "")}
        for p in retrieval.selected_pages
    ]
    confidence = verify.get("confidence") if verify else ("abstain" if path == "none" else "unknown")
    return {
        "run_id": run_id,
        "question": question,
        "answer_path": path,
        "final_answer": final_answer,
        "selected_documents": retrieval.selected_documents,
        "selected_pages": [
            {k: v for k, v in p.items() if k != "text"} | {"text_len": len(p.get("text", ""))}
            for p in retrieval.selected_pages
        ],
        "evidence": evidence,
        "rerank_top_score": retrieval.rerank_top_score,
        "route_reason": retrieval.route_reason,
        "verification": verify,
        "confidence": confidence,
        "rollback_history": history,
        "metrics": m.to_dict(),
    }

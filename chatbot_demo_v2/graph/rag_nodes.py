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

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .state import RagState


@dataclass
class RagDeps:
    """RAG 서브그래프 노드가 쓰는 런타임 의존성(비직렬화 — 상태 대신 여기로)."""
    config: Any
    backend: Any
    prompts: Any = None                           # PromptLoader — grade_evidence 프롬프트용
    scratch: dict = field(default_factory=dict)   # 현재 run 의 metrics(RunMetrics) 등


# ---------------------------------------------------------------------------
# 근거 등급 · 컨텍스트 예산 (2026-07-27 작업 2)
#
# 왜 필요한가 — 실측(LangSmith 8e0815cd):
#   컨텍스트 5,949자 중 정답 근거는 9.6%, 무관한 3순위 페이지가 64.6%, 한 문서가 90.1% 독점.
#   원인은 rag3.answer._format_context 가 페이지 전문을 순위대로 이어붙이며 예산을 **선착순**
#   소진하기 때문. 앞 페이지가 짧으면 뒤의 긴 무관 페이지가 예산을 삼킨다.
#
# 설계 — 관련/무관 이진 컷이 아니라 **3등급 차등**:
#   primary     이 근거만으로 답할 수 있음 → 페이지 전문
#   supporting  주제는 맞으나 핵심은 없음 → 검색이 실제로 고른 청크만 축약 투입
#   irrelevant  제외
#   primary 가 0개여도 버리지 않고 supporting 을 승격한다(그레이딩 오판 대비 안전장치).
# ---------------------------------------------------------------------------
PRIMARY_PAGE_CAP = 4000       # primary 페이지 1장이 쓸 수 있는 최대 글자수
SUPPORTING_PAGE_CAP = 800     # supporting 페이지 1장 상한
SUPPORTING_TOTAL_MAX = 2000   # supporting 전체 합계 절대 상한
#: supporting 합계는 **primary 대비 비율**로도 묶는다.
#: 절대 상한만 두면 primary 가 짧을 때(예 574자) supporting 2,000자가 정답을 묻어버린다.
#: 실측: final_pages 3→6 확대 후 rag_02 의 정답 근거 비중이 61%→45% 로 희석됐다.
SUPPORTING_RATIO_OF_PRIMARY = 0.5
SUPPORTING_TOTAL_MIN = 400    # primary 가 아주 짧아도 최소한의 배경은 남긴다
SUPPORTING_DOC_SHARE = 0.6    # supporting 예산 안에서 한 문서가 차지할 수 있는 최대 비율
GRADE_SNIPPET_CHARS = 320     # 그레이딩 프롬프트에 넣는 페이지당 발췌 길이(비용 억제)


def _retrieval_to_dict(rr) -> dict:
    return {
        "selected_pages": rr.selected_pages,
        "selected_documents": rr.selected_documents,
        "answer_path": rr.answer_path,
        "route_reason": rr.route_reason,
        "rerank_top_score": rr.rerank_top_score,
        # 2026-07-27: 청크별 리랭크 점수는 가장 정밀한 관련도 신호인데 여기서 버려지고 있었다.
        # supporting 등급의 '매칭 청크만 투입'에 필요하다.
        "candidate_chunks": rr.candidate_chunks or [],
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

    # ---------- 청크 본문 조회 (supporting 축약 투입용) ----------
    def _chunk_text_map() -> dict:
        """chunk_id → 색인된 청크 본문. flat 인덱스는 이미 메모리에 로드돼 있어 비용이 없다.

        색인 텍스트는 "{카탈로그 프리픽스} | p{n}\\n{원문}" 형식이라 첫 개행 이후만 취한다.
        """
        cached = deps.scratch.get("_chunk_map")
        if cached is not None:
            return cached
        try:
            from rag3.flat_index import get_flat_chunk_index
            idx = get_flat_chunk_index(config, backend)
            idx._load()  # noqa: SLF001 — 인덱스 자체 API(로드는 멱등)
            out = {}
            for cid, doc in zip(idx._ids, idx._docs):  # noqa: SLF001
                nl = doc.find("\n")
                out[cid] = doc[nl + 1:] if nl != -1 else doc
        except Exception:  # noqa: BLE001 — 조회 실패해도 페이지 텍스트 폴백이 있다
            out = {}
        deps.scratch["_chunk_map"] = out
        return out

    def _matched_chunk_text(page: dict, limit: int) -> str:
        """검색이 실제로 고른 청크 본문만 이어붙인다. 없으면 페이지 텍스트 앞부분으로 폴백."""
        cmap = _chunk_text_map()
        parts, used = [], 0
        for cid in (page.get("matched_chunks") or []):
            t = (cmap.get(cid) or "").strip()
            if not t:
                continue
            room = limit - used
            if room <= 80:
                break
            parts.append(t[:room])
            used += min(len(t), room)
        if parts:
            return "\n".join(parts)
        return (page.get("text") or "")[:limit]

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

        # --- 작업 6: CRAG 재작성 안전가드 ---
        # 재작성이 항상 나은 게 아니다. 실측에서 재작성 질의는 정답 근거를 1위 → 7위로
        # 떨어뜨렸다. 원본 controller_x 는 최소한 answer_path != "none" 검사라도 했지만
        # 서브그래프는 무조건 덮어쓰고 있었다. 더 나쁘면 원본을 유지한다.
        prev = state.get("retrieval") or {}
        old_top = prev.get("rerank_top_score") or 0.0
        new_top = rr.rerank_top_score or 0.0
        if prev and (rr.answer_path == "none" or new_top < old_top):
            _emit("crag", "재작성 결과가 더 낫지 않아 원래 검색 결과를 유지합니다.")
            return {"retrieval": prev,
                    "history": [{"action": "crag_rewrite_rejected",
                                 "old_top": round(old_top, 4), "new_top": round(new_top, 4),
                                 "new_path": rr.answer_path}]}
        return {"retrieval": _retrieval_to_dict(rr),
                "history": [{"action": "crag_rewrite_accepted",
                             "old_top": round(old_top, 4), "new_top": round(new_top, 4)}]}

    # ---------- grade_evidence (S5a, 2026-07-27 신설) ----------
    _GRADES = ("primary", "supporting", "irrelevant")

    def _parse_grades(raw: str, n: int) -> list[str]:
        """'1=primary' 형식을 파싱. 못 읽은 항목은 supporting(중립)으로 둔다 — 조용한 유실 방지."""
        out = ["supporting"] * n
        for m in re.finditer(r"(\d+)\s*[=:.)\-]\s*([A-Za-z_]+)", raw or ""):
            i = int(m.group(1)) - 1
            g = m.group(2).strip().lower()
            if 0 <= i < n and g in _GRADES:
                out[i] = g
        return out

    def grade_evidence(state: RagState) -> dict:
        """회수된 페이지를 primary/supporting/irrelevant 로 판정한다(LLM 1회).

        검색 랭킹만으로는 못 고치는 실패를 여기서 잡는다. 실측에서 리랭커는 "AP 설치"라는
        표현이 겹친다는 이유로 장애조치표를 1순위로 올렸는데, 파라미터 튜닝(프리픽스 제거·
        표 행분할·질의 재작성)으로는 셋 다 오히려 악화됐다. 의미 판정만이 유효했다.
        """
        rr = state["retrieval"]
        pages = rr.get("selected_pages") or []
        if not pages:
            return {"graded_pages": [], "grade_detail": []}

        _emit("grade", f"찾은 근거 {len(pages)}건이 질문에 맞는지 판별하는 중…")
        grades = ["supporting"] * len(pages)
        raw = ""
        if deps.prompts is not None:
            cand = "\n\n".join(
                "[%d] 문서: %s p%s\n%s" % (
                    i + 1, p.get("document_name"), p.get("page_number"),
                    (p.get("text") or "").strip().replace("\n", " ")[:GRADE_SNIPPET_CHARS])
                for i, p in enumerate(pages)
            )
            try:
                prompt = deps.prompts.render("evidence_grader",
                                             question=state["question"], candidates=cand)
                raw = backend.chat_text(prompt) or ""
                metrics.record_judge()
                grades = _parse_grades(raw, len(pages))
            except Exception:  # noqa: BLE001 — 실패 시 전부 supporting(=기존과 유사 동작)
                grades = ["supporting"] * len(pages)

        detail = [
            {"document_name": p.get("document_name"), "page_number": p.get("page_number"),
             "page_score": p.get("page_score"), "grade": g}
            for p, g in zip(pages, grades)
        ]
        kept = [dict(p, _grade=g) for p, g in zip(pages, grades) if g != "irrelevant"]
        n_pri = sum(1 for g in grades if g == "primary")
        _emit("grade", f"근거 판별 완료 — 핵심 {n_pri}건 · 보조 {len(kept) - n_pri}건 "
                       f"· 제외 {len(pages) - len(kept)}건")
        return {"graded_pages": kept, "grade_detail": detail,
                "grade_raw": raw[:400]}

    # ---------- 컨텍스트 예산 배분 ----------
    def _budget_pages(graded: list[dict]) -> list[dict]:
        """등급별 예산을 적용해 **text 를 미리 잘라 둔** 페이지 목록을 만든다.

        rag3.answer._format_context 는 받은 페이지의 text 를 그대로 이어붙이므로,
        여기서 text 를 예산만큼 줄여 두면 ragcore 를 수정하지 않고 예산 배분이 적용된다.
        """
        primary = [p for p in graded if p.get("_grade") == "primary"]
        support = [p for p in graded if p.get("_grade") != "primary"]
        promoted = False
        if not primary and support:
            # primary 0개 → 버리지 않고 최상위 supporting 을 승격(그레이딩 오판 안전장치)
            primary, support, promoted = support[:1], support[1:], True

        out: list[dict] = []
        for p in primary:
            out.append(dict(p, text=(p.get("text") or "")[:PRIMARY_PAGE_CAP]))

        # supporting 총예산 = primary 분량에 비례(절대 상·하한 안에서).
        # 정답 근거가 배경 설명에 묻히지 않게 하는 핵심 장치다.
        primary_chars = sum(len(p.get("text") or "") for p in out)
        sup_total = int(min(SUPPORTING_TOTAL_MAX,
                            max(SUPPORTING_TOTAL_MIN, primary_chars * SUPPORTING_RATIO_OF_PRIMARY)))

        used, by_doc = 0, {}
        doc_cap = int(sup_total * SUPPORTING_DOC_SHARE)
        for p in support:
            room = min(SUPPORTING_PAGE_CAP, sup_total - used,
                       doc_cap - by_doc.get(p.get("document_name"), 0))
            if room <= 120:                     # 남은 예산이 의미 없을 만큼 작으면 건너뛴다
                continue
            txt = _matched_chunk_text(p, room)
            if not txt.strip():
                continue
            out.append(dict(p, text=txt))
            used += len(txt)
            by_doc[p.get("document_name")] = by_doc.get(p.get("document_name"), 0) + len(txt)

        if promoted and out:
            out[0] = dict(out[0], _promoted=True)
        return out

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
        # 2026-07-27: 검색 원본이 아니라 **등급 판정을 통과한 페이지**로 답한다.
        graded = state.get("graded_pages") or rr["selected_pages"]
        pages = _budget_pages(graded) or graded
        path = rr["answer_path"]
        # _route_v2 는 검색 1순위만 보고 경로를 정한다. 1순위가 irrelevant 로 걸러졌으면
        # 실제 답변 근거(pages[0]) 기준으로 경로를 다시 확인한다(ragcore 무수정).
        if path == "vision" and not (pages[0].get("page_type") == "figure"):
            path = "text"
        _emit("answer", f"근거 {len(pages)}페이지로 답변을 생성하는 중…"
              if path == "text" else "표·도표 이미지를 읽어 답변을 생성하는 중…")
        ta = time.time()
        if path == "vision":
            ans = answer_vision_from_page(state["question"], pages[0], backend, config)
        else:
            from rag3x.answer_x import answer_text_from_pages_x
            ans = answer_text_from_pages_x(state["question"], pages, backend, config)
        metrics.record_timing("answer", time.time() - ta)
        return {"answer": ans, "answer_path": path, "budgeted_pages": pages}

    # ---------- verify (S7) ----------
    def verify_node(state: RagState) -> dict:
        rr = state["retrieval"]
        pages = state.get("budgeted_pages") or rr["selected_pages"]
        path = state["answer_path"]
        top = rr.get("rerank_top_score") or 0.0
        _emit("verify", "답변이 근거와 맞는지 검증하는 중…")
        # 단일문서 판정도 실제 답변에 쓰인 페이지 기준으로(등급 통과분).
        used_docs = [{"document_name": d} for d in {p.get("document_name") for p in pages}]
        ground = _decide_ground(state["answer"], path, top, used_docs, config)
        v = _verify(state["answer"], path, pages[0], backend, config, ground=ground)
        return {"verify": v}

    # ---------- rollback A: text 빈응답/확인불가 → 1순위 페이지 단독 재시도 ----------
    def rollback_top1(state: RagState) -> dict:
        _emit("rollback", "답변이 부실해 1순위 근거로 다시 시도하는 중…")
        from rag3x.answer_x import answer_text_from_pages_x
        rr = state["retrieval"]
        # 2026-07-27: 검색 1순위(오답일 수 있음)가 아니라 **등급 통과 1순위**로 재시도한다.
        # 기존에는 pages[:1] 이 무관한 페이지여서 재생성이 구조적으로 더 나빠질 수밖에 없었다.
        pages = state.get("budgeted_pages") or state.get("graded_pages") or rr["selected_pages"]
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
            m.rollback_count = len([h for h in history
                                    if not str(h.get("action", "")).startswith("crag_")])

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
            result["metrics"]["gemini_tokens_think"] = acc.get("think")   # 2026-07-27 추가
            result["metrics"]["gemini_calls"] = acc.get("calls")
            result["metrics"]["gemini_cost"] = round(acc.get("cost", 0.0), 6)
            result["metrics"]["gemini_api_s"] = round(acc.get("api_s", 0.0), 2)
        # 근거 등급 판정 결과를 결과 dict 로 노출(LangSmith·평가 하네스에서 확인용)
        result["evidence_grades"] = state.get("grade_detail") or []
        used = state.get("budgeted_pages") or []
        result["context_pages_used"] = [
            {"document_name": p.get("document_name"), "page_number": p.get("page_number"),
             "grade": p.get("_grade"), "chars": len(p.get("text") or ""),
             "promoted": bool(p.get("_promoted"))}
            for p in used
        ]
        resolve_evidence(result, config)
        return {"result": result}

    # ---------- 라우팅 함수 ----------
    def after_retrieve(state: RagState) -> str:
        rr = state["retrieval"]
        if rr["answer_path"] != "none":
            return "grade"          # 2026-07-27: 답변 전에 근거 등급 판정을 거친다
        top = rr.get("rerank_top_score") or 0.0
        if (getattr(config, "enable_crag", False) and state.get("crag_budget", 0) > 0
                and config.crag_retry_floor <= top < config.rerank_score_floor):
            return "crag"
        return "no_answer"

    def after_grade(state: RagState) -> str:
        """등급 결과로 분기. **primary 가 없어도 버리지 않는 것**이 요점이다.

        - 남은 근거 있음            → answer (primary 0개면 _budget_pages 가 승격)
        - 전부 irrelevant + 예산 有 → crag (의미상 근거가 없을 때 재질의 — 점수 바닥일 때가 아니라)
        - 전부 irrelevant + 예산 無 → no_answer
        """
        kept = state.get("graded_pages") or []
        if kept:
            return "answer"
        if getattr(config, "enable_crag", False) and state.get("crag_budget", 0) > 0:
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
        "grade_evidence": grade_evidence,
        "crag_rewrite": crag_rewrite,
        "answer_node": answer_node,
        "verify_node": verify_node,
        "rollback_top1": rollback_top1,
        "rollback_vision": rollback_vision,
        "rollback_ocr": rollback_ocr,
        "finalize": finalize,
    }
    routers = {"after_retrieve": after_retrieve, "after_grade": after_grade,
               "after_verify": after_verify}
    return nodes, routers

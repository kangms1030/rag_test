"""P3 다문서 종합·추론 — 분해검색 + 문장단위 인용검증.

원본 무수정 재사용: retrieve.run_retrieval(하위질문별 독립 실행), verify.check_claims_supported,
page_store, answer._format_context. 신규 로직만 여기 둔다. controller_x가 라우팅으로만 진입시키며,
단일형 질문은 이 모듈을 타지 않는다(무회귀 보장).

흐름(§P3 c·d):
  라우팅(should_decompose) → 하위질문 2~4개 분해(LLM 1회) → 하위질문별 기존 검색스택 독립 실행
  → 문서별 그룹 컨텍스트 → 합성 답변 1회(문장별 인용 강제) → 문장단위 검증(미지원 문장만 제거,
  전체 abstain 대신 부분답변).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from rag3 import metrics
from rag3.answer import _NO_DOC_ANSWER
from rag3.config import Config
from rag3.models import Backend
from rag3.retrieve import run_retrieval
from rag3.verify import check_claims_supported, is_abstain

logger = logging.getLogger(__name__)

# 종합형 신호 어휘(비교/집계/열거). 하나라도 걸리면 복합 질문 후보.
_COMPARE_TERMS = ["비교", "차이", "각각", "종합", "공통", "대비", "구분",
                  "나누어", "그리고", "및 ", "어떻게 다", "무엇이며", "무엇인가",
                  "각 서비스", "각 앱", "각 항목", "순서", "조합"]


def should_decompose(question: str, config: Config) -> bool:
    """복합(종합형) 질문 라우팅. 플래그 OFF면 항상 False(단일형 경로 무영향)."""
    if not getattr(config, "x_enable_decompose_routing", False):
        return False
    q = question or ""
    hits = sum(1 for t in _COMPARE_TERMS if t in q)
    return hits >= 1


PROMPT_DECOMPOSE = """다음 질문에 정확히 답하려면 문서에서 서로 다른 여러 부분을 찾아 종합해야 한다.
검색에 쓸 하위 질문을 2~4개로 나눠라. 각 하위 질문은 한 줄에 하나씩, 번호·설명 없이 질문만 출력해라.

질문: {question}
하위 질문들:"""


def decompose_question(question: str, backend: Backend, config: Config) -> list[str]:
    raw = backend.chat_text(PROMPT_DECOMPOSE.format(question=question))
    metrics.record_judge()
    subs = []
    for line in (raw or "").splitlines():
        s = line.strip().strip("-*·").strip()
        s = re.sub(r"^\s*\d+[.)]\s*", "", s).strip()  # 앞 번호 제거
        if len(s) >= 6 and not s.startswith(("질문", "하위", "답변")):
            subs.append(s)
    subs = subs[: int(getattr(config, "x_decompose_max_subq", 4))]
    return subs or [question]


def decomposed_retrieval(subquestions: list[str], config: Config, backend: Backend) -> list[dict[str, Any]]:
    """하위질문별 기존 검색스택 독립 실행 → 페이지 병합(문서·페이지 dedup, 점수 최대 보존)."""
    merged: dict[tuple, dict] = {}
    for sq in subquestions:
        r = run_retrieval(sq, config, backend)
        for p in r.selected_pages:
            key = (p["document_name"], p["page_number"])
            if key not in merged or p.get("page_score", 0) > merged[key].get("page_score", 0):
                merged[key] = p
    # 점수순 상위 보존(다문서라 기존 final_pages보다 넉넉히: 2배)
    cap = int(getattr(config, "final_pages", 3)) * 2
    return sorted(merged.values(), key=lambda p: -(p.get("page_score", 0) or 0))[:cap]


def _group_context(pages: list[dict[str, Any]], max_chars: int) -> str:
    """문서별로 묶어 컨텍스트 구성(합성 시 문서 간 대조가 쉽도록)."""
    from collections import defaultdict
    bydoc: dict[str, list] = defaultdict(list)
    for p in pages:
        bydoc[p["document_name"]].append(p)
    parts, used = [], 0
    for doc, ps in bydoc.items():
        block = [f"===== 문서: {doc} ====="]
        for p in ps:
            body = (p.get("text", "") or "")[:1500]
            block.append(f"[p{p['page_number']}] {body}")
        chunk = "\n".join(block)
        if used + len(chunk) > max_chars:
            chunk = chunk[: max(0, max_chars - used)]
        parts.append(chunk)
        used += len(chunk)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


PROMPT_SYNTH = """다음은 여러 문서에서 찾은 근거다. 이 근거만 종합해 질문에 답해라.
규칙:
- 각 문장 끝에 근거 페이지를 [p숫자] 형식으로 표기해라(예: ...제공한다[p11]).
- 근거에 없는 수치·사실은 쓰지 마라. 비교/집계 질문이면 항목별로 정리해라.

질문: {question}

근거:
{context}

답변(문장마다 [p숫자] 표기):"""

_SENT_SPLIT = re.compile(r"(?<=[.。!?])\s+|\n+")


def _sentence_citation_filter(answer: str, context: str) -> tuple[str, list[str]]:
    """문장단위 검증: 미지원 수치/코드를 담은 문장만 제거(부분답변). 제거 문장 리스트 반환."""
    kept, dropped = [], []
    for sent in _SENT_SPLIT.split(answer):
        s = sent.strip()
        if not s:
            continue
        unsupported = check_claims_supported(s, context)
        if unsupported:
            dropped.append(s)
        else:
            kept.append(s)
    return (" ".join(kept).strip(), dropped)


def decompose_answer(question: str, config: Config, backend: Backend) -> dict[str, Any]:
    """분해검색 → 합성 → 문장검증. controller_x가 호출. 반환은 answer_text_from_pages와 호환 dict + 확장."""
    subs = decompose_question(question, backend, config)
    pages = decomposed_retrieval(subs, config, backend)
    if not pages:
        return {"final_answer": _NO_DOC_ANSWER, "raw": "", "context": "",
                "selected_pages": [], "subquestions": subs, "answer_path": "none"}
    context = _group_context(pages, int(getattr(config, "context_max_chars", 10000)))
    raw = backend.chat_text(PROMPT_SYNTH.format(question=question, context=context))
    metrics.record_text_answer()
    final = raw.strip()
    dropped: list[str] = []
    if getattr(config, "x_sentence_citation_verify", False):
        filtered, dropped = _sentence_citation_filter(raw, context)
        # 부분답변이 비면(전 문장 미지원) 원답 유지보다 정직 abstain
        final = filtered or _NO_DOC_ANSWER
    return {
        "final_answer": final or _NO_DOC_ANSWER,
        "raw": raw,
        "context": context,
        "selected_pages": pages,
        "subquestions": subs,
        "dropped_sentences": dropped,
        "answer_path": "text",
    }

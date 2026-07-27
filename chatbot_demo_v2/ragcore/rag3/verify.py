"""답변 검증 (Phase 2, S7). 싼 것부터: 결정론 숫자/시리얼 대조 -> (vision) 전사-OCR diff -> groundedness LLM.

Phase 0-A/fact: VLM 전사는 숫자·시리얼·용어를 오독한다. 답변이 근거에 실재하지 않는 수치/코드를
주장하면 결정론적으로 잡아낸다(모델 호출 0). groundedness는 12b 짧은 호출(Phase 0-E: 12b 겸용,
스왑 0). LLM 판정 단독으로는 롤백을 유발하지 않고 신뢰도 태그만 부여한다(계획 §무한루프 규칙 4).
"""
from __future__ import annotations

import logging
import re
from typing import Any

from . import metrics
from .config import Config
from .models import Backend

logger = logging.getLogger(__name__)

# 숫자(콤마/소수 허용), 대문자+숫자 시리얼/코드, 전화번호
_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_CODE_RE = re.compile(r"\b(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{4,}\b")

_ABSTAIN_MARKERS = ("확인 불가", "확인할 수 없", "제공된 근거", "찾을 수 없", "알 수 없")

PROMPT_GROUND = """아래 [근거]만 보고 [답변]이 근거로 뒷받침되는지 판정해라.
- 답변의 모든 수치·사실이 근거에서 확인되면: SUPPORTED
- 답변이 근거에 없는 수치·사실을 주장하면: NOT_SUPPORTED
- 답변이 "확인 불가"류로 답을 회피하면: ABSTAIN
반드시 위 셋 중 한 단어로만 답해라.

[근거]
{context}

[답변]
{answer}

판정:"""


def _norm_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")


def extract_claims(text: str) -> tuple[set[str], set[str]]:
    nums = {m.group() for m in _NUM_RE.finditer(text)}
    codes = {m.group() for m in _CODE_RE.finditer(text)}
    return nums, codes


def check_claims_supported(answer: str, context: str) -> list[str]:
    """답변의 숫자/코드 토큰 중 근거 컨텍스트에 실재하지 않는 것(=미지원 주장) 리스트."""
    a_nums, a_codes = extract_claims(answer)
    ctx = context
    ctx_nc = context.replace(",", "")
    unsupported: list[str] = []
    for n in a_nums:
        digits = _norm_num(n).replace(".", "")
        if len(digits) < 3:  # 한두 자리(단계·순번 등)는 노이즈 -> 스킵
            continue
        if n in ctx or _norm_num(n) in ctx_nc:
            continue
        unsupported.append(n)
    for c in a_codes:
        if c in ctx:
            continue
        unsupported.append(c)
    return unsupported


def is_abstain(answer: str) -> bool:
    a = (answer or "").strip()
    return (a == "") or any(mk in a for mk in _ABSTAIN_MARKERS)


def groundedness(answer: str, context: str, backend: Backend, config: Config) -> str:
    """12b 짧은 호출로 supported/not_supported/abstain 판정."""
    model = config.verify_model or config.text_answer_model
    prompt = PROMPT_GROUND.format(context=context[: config.context_max_chars], answer=answer[:1500])
    raw = backend.chat_text(prompt, model=model).strip().upper()
    metrics.record_verify()
    if "NOT_SUPPORTED" in raw or "NOT SUPPORTED" in raw:
        return "not_supported"
    if "ABSTAIN" in raw:
        return "abstain"
    if "SUPPORTED" in raw:
        return "supported"
    return "unknown"


def verify_answer(
    answer: str,
    context: str,
    *,
    answer_path: str,
    transcription: str | None,
    ocr_text: str | None,
    backend: Backend,
    config: Config,
    run_groundedness: bool = True,
) -> dict[str, Any]:
    """검증 결과 dict. 결정론 신호(unsupported/abstain/ocr_mismatch)는 롤백 트리거, groundedness는 태그."""
    result: dict[str, Any] = {
        "abstain": is_abstain(answer),
        "unsupported_claims": check_claims_supported(answer, context),
    }
    # vision: 전사 vs MinerU OCR 텍스트 불일치(전사가 OCR에 없는 수치/코드)
    if answer_path == "vision" and transcription and ocr_text:
        result["transcription_ocr_mismatch"] = check_claims_supported(transcription, ocr_text)
    else:
        result["transcription_ocr_mismatch"] = []

    result["deterministic_ok"] = (
        not result["abstain"]
        and not result["unsupported_claims"]
        and not result["transcription_ocr_mismatch"]
    )

    if run_groundedness and not result["abstain"]:
        result["groundedness"] = groundedness(answer, context, backend, config)
    else:
        result["groundedness"] = "skipped"
    # 신뢰도 태그: 결정론 통과 + groundedness supported면 high
    if result["deterministic_ok"] and result["groundedness"] in ("supported", "skipped"):
        result["confidence"] = "high"
    elif result["abstain"]:
        result["confidence"] = "abstain"
    else:
        result["confidence"] = "low"
    return result

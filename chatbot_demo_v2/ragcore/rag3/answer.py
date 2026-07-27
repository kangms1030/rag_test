"""텍스트/비전 단일호출 답변 생성.

test_1차와 달리 별도 verify 호출이 없다 — 표 질문은 "표 셀을 먼저 그대로 옮겨 적고
그 값만으로 답하라"는 단일 프롬프트(transcribe-then-answer)로 오독을 억제한다
(숫자 정확도 3중 대책의 ③, 설계 문서 참고).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import metrics
from .config import Config
from .models import Backend
from .retrieve import RetrievalResult

logger = logging.getLogger(__name__)

_NO_DOC_ANSWER = "선택된 문서에서 확인 불가"


def _looks_starved(raw: str, top_text: str = "") -> bool:
    """12b가 답을 굶은 정황(완전 빈 응답 또는 제목/헤딩만 뱉고 조기 종료).

    P0-C: 엄격 프롬프트 + 노이즈 표 컨텍스트에서 본문 생성 전에 EOS로 끊긴다. 정상 abstain
    문구('확인할 수 없습니다')는 제외한다(그건 재시도해도 동일).
    """
    s = raw.strip().strip("'\"`*#·-—> ")  # 모델이 덧붙이는 인용부호/마크다운 기호 제거
    if not s:
        return True
    if "확인" in s or "없" in s:  # 정상 abstain 문구는 굶음 아님
        return False
    # 제목 에코: 답변이 근거 첫 부분(헤딩)에 그대로 들어있으면 본문 생성 실패로 본다.
    # (len<40 같은 blanket 규칙은 정상적인 짧은 답변을 오탐해 오히려 손해라 쓰지 않는다.)
    if top_text and len(s) < 60 and s.replace(" ", "") in top_text[:150].replace(" ", ""):
        return True
    return False

PROMPT_TEXT_ANSWER = """다음은 질문에 대한 근거 문서 페이지의 텍스트(표는 마크다운으로 포함)다.
이 내용만 근거로 질문에 답해줘. 근거에 없는 내용은 추측하지 말고 "제공된 근거에서 확인할 수 없습니다"라고 답해.
표에 숫자가 있으면 표의 행/열을 정확히 대조해서 답해.

질문: {question}

근거:
{context}

답변:"""

PROMPT_TEXT_ANSWER_SIMPLE = """다음 근거만 사용해 질문에 간단히 답해줘. 근거에 정말 없으면 "제공된 근거에서 확인할 수 없습니다"라고만 답해.

질문: {question}

근거:
{context}

답변:"""

PROMPT_VISION_ANSWER = """다음 이미지는 질문에 대한 근거 문서의 표/도표 페이지다.

절차:
1. 먼저 이미지에서 질문과 관련된 표의 행/열(또는 도표 수치)을 그대로 옮겨 적어라(전사).
2. 전사한 내용만 근거로 질문에 답하라. 이미지에 없는 내용은 추측하지 마라.

질문: {question}

반드시 아래 형식으로 답해:
[전사]
(관련 표/도표 내용을 그대로 옮겨 적기)

[답변]
(전사한 내용만 근거로 한 답변)
"""


def _format_context(pages: list[dict[str, Any]], max_chars: int | None = None) -> str:
    """페이지 텍스트를 헤더와 함께 이어붙인다. max_chars가 주어지면 상위 페이지부터 예산 내로 트림.

    Phase 0-C: num_ctx 8192에서 큰 표HTML 컨텍스트가 출력 굶김으로 빈 응답을 유발했다.
    num_ctx는 16384로 올리되(config), 컨텍스트도 예산 상한으로 트림해 이중 방어한다.
    """
    parts = []
    used = 0
    for p in pages:
        header = f"--- {p['document_name']} p{p['page_number']} ---"
        body = p.get("text", "")
        block = f"{header}\n{body}"
        if max_chars is not None and used + len(block) > max_chars:
            remaining = max(0, max_chars - used - len(header) - 1)
            if remaining < 200:  # 남은 예산이 너무 작으면 이 페이지는 생략
                break
            block = f"{header}\n{body[:remaining]}"
            parts.append(block)
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def extract_transcription(vision_raw: str) -> str:
    """[전사]와 [답변] 사이(또는 [전사] 이후 [답변] 전까지)를 전사 텍스트로 추출."""
    t = "[전사]"
    a = "[답변]"
    ti = vision_raw.find(t)
    ai = vision_raw.find(a)
    if ti == -1:
        return ""
    start = ti + len(t)
    end = ai if ai != -1 else len(vision_raw)
    return vision_raw[start:end].strip()


def _extract_final_answer(vision_raw: str) -> str:
    """[답변] 이후만 최종 답변으로 남기고, 없으면 전문을 그대로 둔다(전사 과정도 근거로 유용)."""
    marker = "[답변]"
    idx = vision_raw.find(marker)
    if idx == -1:
        return vision_raw.strip()
    return vision_raw[idx + len(marker) :].strip()


def answer_text_from_pages(question: str, pages: list[dict[str, Any]], backend: Backend, config: Config) -> dict[str, Any]:
    """지정 페이지들로 text 답변 1회. {final_answer, raw, context} 반환."""
    context = _format_context(pages, max_chars=config.context_max_chars)
    prompt = PROMPT_TEXT_ANSWER.format(question=question, context=context)
    raw = backend.chat_text(prompt)
    metrics.record_text_answer()
    # P0-C: 여러 페이지를 이어붙인 컨텍스트에 지저분한 표 HTML이 섞이면 12b가 본문 생성 전
    # EOS로 끊겨 빈 응답/제목만 뱉는다(굶음). 정답은 대개 1순위 페이지에 있으므로, 굶으면
    # **1순위 페이지 단독 + 간결 프롬프트**로 1회 재생성한다(노이즈 페이지 제거가 핵심).
    if pages and _looks_starved(raw, pages[0].get("text", "")):
        ctx1 = _format_context(pages[:1], max_chars=config.context_max_chars)
        retry = backend.chat_text(PROMPT_TEXT_ANSWER_SIMPLE.format(question=question, context=ctx1))
        metrics.record_text_answer()
        # 안전장치: 재시도가 더 풍부할 때만 채택(원답이 멀쩡한데 top-1로 정보 손실하는 경우 방지)
        if len(retry.strip()) > len(raw.strip()):
            raw = retry
    return {"final_answer": raw.strip() or _NO_DOC_ANSWER, "raw": raw, "context": context}


def resolve_cached_path(stored: str, config: Config) -> Path | None:
    """저장된 이미지 경로가 폴더 재구성으로 stale일 수 있어(예: 구 test_2\\rag2\\cache\\...),
    실제 파일을 source_parsed 기준으로 재배치해 찾는다. 없으면 None."""
    if not stored:
        return None
    p = Path(stored)
    if p.exists():
        return p
    parts = p.parts
    if "parsed" in parts:
        tail = Path(*parts[parts.index("parsed") + 1:])
        cand = config.source_parsed / tail
        if cand.exists():
            return cand
    return None


def answer_vision_from_page(question: str, page: dict[str, Any], backend: Backend, config: Config) -> dict[str, Any]:
    """1개 페이지 이미지(표 크롭 우선)로 vision 답변 1회. {final_answer, raw, transcription, context} 반환.

    이미지 파일이 없으면(경로 stale/삭제) text 경로로 안전 폴백한다.
    """
    crop = resolve_cached_path(page.get("table_crop_path", ""), config)
    pageimg = resolve_cached_path(page.get("page_image_path", ""), config)
    image_path = crop or pageimg
    if image_path is None:
        logger.warning("vision 이미지 없음(p%s) -> text 폴백", page.get("page_number"))
        out = answer_text_from_pages(question, [page], backend, config)
        out["transcription"] = ""
        return out
    prompt = PROMPT_VISION_ANSWER.format(question=question)
    raw = backend.chat_vision_text(prompt, [image_path])
    metrics.record_vision_answer()
    return {
        "final_answer": _extract_final_answer(raw) or _NO_DOC_ANSWER,
        "raw": raw,
        "transcription": extract_transcription(raw),
        "context": page.get("text", ""),  # 검증용 OCR/표 텍스트
    }


def generate_answer(question: str, retrieval: RetrievalResult, backend: Backend, config: Config) -> dict[str, Any]:
    """{"final_answer", "answer_path", "raw_model_output", "evidence"} 반환."""
    if retrieval.answer_path == "none" or not retrieval.selected_pages:
        return {
            "final_answer": _NO_DOC_ANSWER,
            "answer_path": "none",
            "raw_model_output": "",
            "evidence": [],
            "skip_reason": retrieval.route_reason,
        }

    pages = retrieval.selected_pages
    evidence = [
        {
            "document_name": p["document_name"],
            "page_number": p["page_number"],
            "page_image_path": p["page_image_path"],
            "table_crop_path": p.get("table_crop_path", ""),
        }
        for p in pages
    ]

    if retrieval.answer_path == "vision":
        best = pages[0]
        image_path = Path(best["table_crop_path"]) if best.get("table_crop_path") else Path(best["page_image_path"])
        prompt = PROMPT_VISION_ANSWER.format(question=question)
        raw = backend.chat_vision_text(prompt, [image_path])
        metrics.record_vision_answer()
        final_answer = _extract_final_answer(raw) or _NO_DOC_ANSWER
        return {"final_answer": final_answer, "answer_path": "vision", "raw_model_output": raw, "evidence": evidence}

    context = _format_context(pages)
    prompt = PROMPT_TEXT_ANSWER.format(question=question, context=context)
    raw = backend.chat_text(prompt)
    metrics.record_text_answer()
    final_answer = raw.strip() or _NO_DOC_ANSWER
    return {"final_answer": final_answer, "answer_path": "text", "raw_model_output": raw, "evidence": evidence}

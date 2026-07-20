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

PROMPT_TEXT_ANSWER = """다음은 질문에 대한 근거 문서 페이지의 텍스트(표는 마크다운으로 포함)다.
이 내용만 근거로 질문에 답해줘. 근거에 없는 내용은 추측하지 말고 "제공된 근거에서 확인할 수 없습니다"라고 답해.
표에 숫자가 있으면 표의 행/열을 정확히 대조해서 답해.

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


def _format_context(pages: list[dict[str, Any]]) -> str:
    parts = []
    for p in pages:
        header = f"--- {p['document_name']} p{p['page_number']} ---"
        parts.append(f"{header}\n{p['text']}")
    return "\n\n".join(parts)


def _extract_final_answer(vision_raw: str) -> str:
    """[답변] 이후만 최종 답변으로 남기고, 없으면 전문을 그대로 둔다(전사 과정도 근거로 유용)."""
    marker = "[답변]"
    idx = vision_raw.find(marker)
    if idx == -1:
        return vision_raw.strip()
    return vision_raw[idx + len(marker) :].strip()


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

"""CRAG식 질의 재작성 (Phase 2, S3a). 검색 게이트가 거절(none)했으나 완전 무관은 아닌 경계 질문에서
핵심 키워드 중심으로 질의를 1회 재작성해 재검색한다. 모델 호출 1회(12b), controller가 1회로 제한한다.
"""
from __future__ import annotations

import logging

from . import metrics
from .config import Config
from .models import Backend

logger = logging.getLogger(__name__)

PROMPT_REWRITE = """다음 질문으로 문서를 검색했으나 관련 근거를 충분히 찾지 못했다.
문서 검색(키워드/의미)에 더 잘 걸리도록 핵심 개체·용어 중심으로 질문을 1개만 다시 써라.
설명 없이 재작성된 질문 문장만 출력해라.

원 질문: {question}
재작성된 질문:"""


def rewrite_query(question: str, backend: Backend, config: Config) -> str:
    model = config.verify_model or config.text_answer_model
    raw = backend.chat_text(PROMPT_REWRITE.format(question=question), model=model)
    metrics.record_judge()
    line = (raw or "").strip().splitlines()[0] if raw.strip() else question
    return line.strip().strip('"').strip()[:300] or question

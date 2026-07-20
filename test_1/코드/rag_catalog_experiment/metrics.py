"""캐시 독립적인 실험 비용 지표: contextvar 누산기.

`Backend`를 감싸는 프록시가 아니라 결정 지점(`get_or_build_summary`/`get_or_build_chunks`/
`analyze_query`/`generate_answer`/`verify_answer`/`HybridIndex.query`/`upsert`)에서 직접
기록한다. 두 가지 이유가 있다:

1. 디스크 캐시 히트는 backend를 아예 호출하지 않으므로, 프록시로는 "이 모드가 요약하기로
   결정한 (문서,페이지) 수"를 셀 수 없다 — 그런데 그게 카탈로그 비용 비교의 핵심 지표다.
2. `HybridIndex`는 `backend.backend_id`로 Chroma 디렉터리를 정하고 `id(backend)`로 인덱스를
   캐시한다(`indexes.py`). run마다 새 프록시로 backend를 감싸면 색인이 빈 디렉터리를
   가리키거나 같은 경로에 `PersistentClient`가 여러 개 열려 손상 위험이 생긴다.

contextvar이므로 활성 run이 없으면(= `run_metrics()` 밖) 모든 `record_*`는 조용히 no-op이다.
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_LLM_STAGES = ("answer", "verify", "query_analyzer", "embed")


@dataclass
class RunMetrics:
    mode: str = "catalog"
    depth: str = "answer"
    is_ingest: bool = False

    # 캐시 독립적 비용 지표 — 중복 제거된 (doc_slug, page_number) 집합.
    # 비용 비교 결론은 이 두 값(의 len)으로 낸다. run 순서/캐시 온도에 흔들리지 않는다.
    summary_pages_required: set[tuple[str, int]] = field(default_factory=set)
    chunk_pages_required: set[tuple[str, int]] = field(default_factory=set)

    # 캐시 의존적 실측 호출 수 — 참고용. 캐시가 warm하면 0에 가까워진다.
    summary_calls: int = 0
    chunk_calls: int = 0
    answer_vlm_calls: int = 0
    verify_llm_calls: int = 0
    query_analyzer_calls: int = 0
    embed_calls: int = 0

    # page_index 전체 커버리지 스냅샷 (retrieval.py가 run 시작 시 채움).
    # 969쪽 중 몇 쪽이 이미 요약되어 있었는지 — no_catalog 해석에 필수.
    page_index_coverage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "depth": self.depth,
            "is_ingest": self.is_ingest,
            "summary_pages_required": len(self.summary_pages_required),
            "chunk_pages_required": len(self.chunk_pages_required),
            "summary_calls": self.summary_calls,
            "chunk_calls": self.chunk_calls,
            "answer_vlm_calls": self.answer_vlm_calls,
            "verify_llm_calls": self.verify_llm_calls,
            "query_analyzer_calls": self.query_analyzer_calls,
            "embed_calls": self.embed_calls,
            "page_index_coverage": self.page_index_coverage,
        }


_CURRENT: contextvars.ContextVar[RunMetrics | None] = contextvars.ContextVar("run_metrics", default=None)


@contextmanager
def run_metrics(m: RunMetrics) -> Iterator[RunMetrics]:
    token = _CURRENT.set(m)
    try:
        yield m
    finally:
        _CURRENT.reset(token)


def current() -> RunMetrics | None:
    return _CURRENT.get()


def record_summary(doc_slug: str, page_number: int, *, was_cached: bool) -> None:
    m = _CURRENT.get()
    if m is None:
        return
    m.summary_pages_required.add((doc_slug, page_number))
    if not was_cached:
        m.summary_calls += 1


def record_chunk(doc_slug: str, page_number: int, *, was_cached: bool) -> None:
    m = _CURRENT.get()
    if m is None:
        return
    m.chunk_pages_required.add((doc_slug, page_number))
    if not was_cached:
        m.chunk_calls += 1


def record_llm(stage: str) -> None:
    m = _CURRENT.get()
    if m is None:
        return
    if stage not in _LLM_STAGES:
        logger.warning("알 수 없는 metrics stage: %s", stage)
        return
    attr = {
        "answer": "answer_vlm_calls",
        "verify": "verify_llm_calls",
        "query_analyzer": "query_analyzer_calls",
        "embed": "embed_calls",
    }[stage]
    setattr(m, attr, getattr(m, attr) + 1)


def set_page_index_coverage(coverage: dict[str, Any]) -> None:
    m = _CURRENT.get()
    if m is None:
        return
    m.page_index_coverage = coverage

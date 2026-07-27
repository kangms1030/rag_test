"""질문 1건의 단계별 소요시간·모델 호출수 집계 (contextvar 누산기).

test_1차와 달리 캐시 히트/lazy 생성 구분이 필요 없다 — ingest 때 전부 선연산되므로
ask 경로의 호출수는 항상 "임베딩 1회 + 답변 1회"에 수렴해야 하고, 이 모듈은 그걸
실측으로 검증하기 위한 카운터일 뿐이다.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class RunMetrics:
    embed_calls: int = 0
    text_answer_calls: int = 0
    vision_answer_calls: int = 0
    rerank_calls: int = 0
    judge_calls: int = 0
    verify_calls: int = 0
    rollback_count: int = 0
    #: 콜드스타트 whitespace 런어웨이(done_reason=length) 감지로 동일 호출을 재발행한 횟수.
    #: 논리적 모델 호출이 아니라 "내부 복구"이므로 total_model_calls에는 넣지 않고 별도 계측만 한다.
    length_retry_count: int = 0

    timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embed_calls": self.embed_calls,
            "text_answer_calls": self.text_answer_calls,
            "vision_answer_calls": self.vision_answer_calls,
            "rerank_calls": self.rerank_calls,
            "judge_calls": self.judge_calls,
            "verify_calls": self.verify_calls,
            "rollback_count": self.rollback_count,
            "length_retry_count": self.length_retry_count,
            "total_model_calls": self.embed_calls + self.text_answer_calls + self.vision_answer_calls,
            "timings_seconds": self.timings,
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


def record_embed() -> None:
    m = _CURRENT.get()
    if m is not None:
        m.embed_calls += 1


def record_text_answer() -> None:
    m = _CURRENT.get()
    if m is not None:
        m.text_answer_calls += 1


def record_vision_answer() -> None:
    m = _CURRENT.get()
    if m is not None:
        m.vision_answer_calls += 1


def record_rerank() -> None:
    m = _CURRENT.get()
    if m is not None:
        m.rerank_calls += 1


def record_judge() -> None:
    m = _CURRENT.get()
    if m is not None:
        m.judge_calls += 1


def record_verify() -> None:
    m = _CURRENT.get()
    if m is not None:
        m.verify_calls += 1


def record_length_retry() -> None:
    m = _CURRENT.get()
    if m is not None:
        m.length_retry_count += 1


def record_timing(stage: str, seconds: float) -> None:
    m = _CURRENT.get()
    if m is not None:
        m.timings[stage] = round(seconds, 3)

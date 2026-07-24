"""텍스트 정규화 + 결정론적 유사도 매칭 (LLM 미사용)."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz, process

from .models import FaqStore, MatchResult

_WS_RE = re.compile(r"\s+")
# 앞뒤 및 반복 구두점 제거용(문장부호 차이로 인한 오탐 방지)
_TRIM_PUNCT = " \t\r\n?!.,~·…\"'()[]{}"


def normalize_text(s: str) -> str:
    """유사도 비교/정확 일치를 위한 공용 정규화.

    - NFKC 유니코드 정규화
    - 소문자화(영문)
    - 줄바꿈/반복 공백 → 단일 공백
    - 앞뒤 공백·구두점 정리
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = _WS_RE.sub(" ", s)
    s = s.strip().strip(_TRIM_PUNCT).strip()
    return s.lower()


class ScenarioMatcher:
    """자유 입력 질문을 모범 질답과 엄격하게 비교한다.

    절차: 정규화 → 정확 일치(dict) → RapidFuzz fuzz.ratio 상위 2건 →
    best >= threshold AND (best-second) >= margin 이면 채택.
    점수는 0.0~1.0 스케일.
    """

    def __init__(self, faq: FaqStore, threshold: float, margin: float):
        self._faq = faq
        self._threshold = float(threshold)
        self._margin = float(margin)
        self._choices = faq.normalized_choices()  # norm_q -> FaqEntry
        self._keys = list(self._choices.keys())

    def match(self, normalized_question: str) -> MatchResult:
        q = normalized_question or ""

        # 1) 정확 일치
        exact_entry = self._faq.exact(q)
        if exact_entry is not None:
            return MatchResult(
                decision="exact",
                decision_reason="정규화된 질문이 모범 질문과 완전 일치",
                best_score=1.0,
                second_score=0.0,
                margin_observed=1.0,
                threshold=self._threshold,
                margin_required=self._margin,
                matched_id=exact_entry.id,
                matched_question=exact_entry.question,
                matched_sheet=exact_entry.sheet,
                matched_row=exact_entry.row,
            )

        if not self._keys or not q:
            return MatchResult(
                decision="reject_low_score",
                decision_reason="후보 없음 또는 빈 입력",
                best_score=0.0,
                second_score=0.0,
                margin_observed=0.0,
                threshold=self._threshold,
                margin_required=self._margin,
            )

        # 2) RapidFuzz 상위 2건 (fuzz.ratio: 순서 민감, 엄격)
        results = process.extract(
            q, self._keys, scorer=fuzz.ratio, limit=2
        )
        best_key, best_raw, _ = results[0]
        best = best_raw / 100.0
        second = (results[1][1] / 100.0) if len(results) > 1 else 0.0
        margin_obs = best - second
        best_entry = self._choices[best_key]

        base = dict(
            best_score=best,
            second_score=second,
            margin_observed=margin_obs,
            threshold=self._threshold,
            margin_required=self._margin,
            matched_id=best_entry.id,
            matched_question=best_entry.question,
            matched_sheet=best_entry.sheet,
            matched_row=best_entry.row,
        )

        if best < self._threshold:
            return MatchResult(
                decision="reject_low_score",
                decision_reason=(
                    f"최고 점수 {best:.3f} < 임계값 {self._threshold:.3f}"
                ),
                **base,
            )
        if margin_obs < self._margin:
            return MatchResult(
                decision="reject_ambiguous",
                decision_reason=(
                    f"1~2위 점수 차 {margin_obs:.3f} < 여유 {self._margin:.3f} (애매)"
                ),
                **base,
            )
        return MatchResult(
            decision="accept",
            decision_reason=(
                f"최고 점수 {best:.3f} >= {self._threshold:.3f}, "
                f"여유 {margin_obs:.3f} >= {self._margin:.3f}"
            ),
            **base,
        )

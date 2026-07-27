"""시나리오/FAQ 도메인 모델 (직렬화 가능한 dataclass)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FaqEntry:
    """엑셀 모범 질답 한 행."""

    id: str                       # "시트명:행번호"
    sheet: str
    row: int                      # 1-based 엑셀 행 번호
    no: Optional[int]             # 시트 내 No 컬럼
    question_type: Optional[str]  # 질문 유형 (빈 셀이면 None)
    fault_type: Optional[str]     # 장애 유형
    question: str                 # 원문 질문
    question_normalized: str      # 정규화된 질문(유사도 비교용)
    answer: str                   # 원문 답변 (그대로 반환)
    source_files: list[str] = field(default_factory=list)  # 근거 파일명들


class FaqStore:
    """FaqEntry 모음. 정확 일치 조회와 유사도 후보 목록을 제공한다."""

    def __init__(self, entries: list[FaqEntry]):
        self._entries = list(entries)
        self._by_id = {e.id: e for e in self._entries}
        # 정규화 질문 → entry (동일 정규화 질문이 여러 개면 첫 항목 우선)
        self._by_norm: dict[str, FaqEntry] = {}
        for e in self._entries:
            self._by_norm.setdefault(e.question_normalized, e)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[FaqEntry]:
        return list(self._entries)

    def get(self, entry_id: str) -> Optional[FaqEntry]:
        return self._by_id.get(entry_id)

    def get_by_sheet_row(self, sheet: str, row: int) -> Optional[FaqEntry]:
        return self._by_id.get(f"{sheet}:{row}")

    def exact(self, normalized_question: str) -> Optional[FaqEntry]:
        return self._by_norm.get(normalized_question)

    def normalized_choices(self) -> dict[str, FaqEntry]:
        """정규화 질문 → entry 매핑(유사도 검색 후보)."""
        return dict(self._by_norm)


@dataclass(frozen=True)
class MatchResult:
    """유사도 매칭 결과(결정론적)."""

    decision: str          # "exact" | "accept" | "reject_low_score" | "reject_ambiguous"
    decision_reason: str
    best_score: float      # 0.0 ~ 1.0
    second_score: float
    margin_observed: float
    threshold: float
    margin_required: float
    matched_id: Optional[str] = None
    matched_question: Optional[str] = None
    matched_sheet: Optional[str] = None
    matched_row: Optional[int] = None

    @property
    def accepted(self) -> bool:
        return self.decision in ("exact", "accept")

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "best_score": self.best_score,
            "second_score": self.second_score,
            "margin_observed": self.margin_observed,
            "threshold": self.threshold,
            "margin_required": self.margin_required,
            "matched_id": self.matched_id,
            "matched_question": self.matched_question,
            "matched_sheet": self.matched_sheet,
            "matched_row": self.matched_row,
        }


@dataclass(frozen=True)
class ScenarioOption:
    option_id: str
    label: str
    next_node_id: str

    def to_button(self, scenario_id: str, node_id: str) -> dict:
        """프론트로 보낼 버튼 payload."""
        return {
            "scenario_id": scenario_id,
            "node_id": node_id,
            "option_id": self.option_id,
            "label": self.label,
        }


@dataclass(frozen=True)
class ScenarioNode:
    node_id: str
    scenario_id: str
    node_type: str                 # "menu" | "question" | "terminal"
    text: Optional[str]
    options: list[ScenarioOption] = field(default_factory=list)
    # terminal 노드에서 로더가 해석해 채운다.
    answer_text: Optional[str] = None
    answer_source: Optional[str] = None  # "scenario_ppt" | "faq_ref"
    answer_ref_sheet: Optional[str] = None
    answer_ref_row: Optional[int] = None

    @property
    def is_terminal(self) -> bool:
        return self.node_type == "terminal"

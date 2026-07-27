"""FastAPI 요청/응답 스키마(pydantic).

v1 대비 확장(Phase 3~4): clarify_response 입력, 응답에 type/run_id/clarify/
original_answer/faq_evidence/composed/grader_verdict.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScenarioAction(BaseModel):
    type: str = Field(..., description="scenario_option")
    scenario_id: Optional[str] = None
    node_id: str
    option_id: str
    label: Optional[str] = None


class ClarifyResponse(BaseModel):
    """clarify 되묻기에 대한 사용자 선택. choice = faq_id | '__none__'."""
    choice: str


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    action: Optional[ScenarioAction] = None
    clarify_response: Optional[ClarifyResponse] = None


class ResetRequest(BaseModel):
    session_id: str


class WarmupRequest(BaseModel):
    deep: Optional[bool] = None


class FeedbackRequest(BaseModel):
    run_id: str
    score: int          # 1(👍) | 0(👎)
    comment: Optional[str] = None


class ScenarioBlock(BaseModel):
    scenario_id: Optional[str] = None
    node_id: Optional[str] = None
    completed: bool = False


class ChatResponse(BaseModel):
    session_id: str
    type: str = "answer"            # "answer" | "clarify"
    run_id: Optional[str] = None
    route: Optional[str] = None
    answer: Optional[str] = None
    options: list[dict] = []
    scenario: ScenarioBlock = ScenarioBlock()
    confidence: Optional[str] = None
    answer_path: Optional[str] = None
    answer_source: Optional[str] = None
    evidence: list[dict] = []
    faq_evidence: list[dict] = []
    verification: Optional[dict] = None
    source_meta: Optional[dict] = None
    trace: list[dict] = []
    timings: dict = {}
    elapsed_seconds: float = 0.0
    scenario_match: Optional[dict] = None
    warnings: list[str] = []
    # v2 신규
    clarify: Optional[dict] = None        # {candidates: [{faq_id, question, score}]}
    original_answer: Optional[str] = None  # FAQ 합성 시 원문
    composed: bool = False
    grader_verdict: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    engine: dict = {}
    langsmith: dict = {}
    web_search: dict = {}
    routing: dict = {}
    toggles: dict = {}
    graph_mermaid: Optional[str] = None

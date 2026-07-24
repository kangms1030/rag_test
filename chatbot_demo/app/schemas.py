"""FastAPI 요청/응답 스키마(pydantic)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ScenarioAction(BaseModel):
    type: str = Field(..., description="scenario_option")
    scenario_id: Optional[str] = None
    node_id: str
    option_id: str
    label: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    action: Optional[ScenarioAction] = None


class ResetRequest(BaseModel):
    session_id: str


class WarmupRequest(BaseModel):
    deep: Optional[bool] = None


class ScenarioBlock(BaseModel):
    scenario_id: Optional[str] = None
    node_id: Optional[str] = None
    completed: bool = False


class ChatResponse(BaseModel):
    session_id: str
    route: Optional[str] = None
    answer: Optional[str] = None
    options: list[dict] = []
    scenario: ScenarioBlock = ScenarioBlock()
    confidence: Optional[str] = None
    answer_path: Optional[str] = None
    answer_source: Optional[str] = None
    evidence: list[dict] = []
    verification: Optional[dict] = None
    source_meta: Optional[dict] = None
    trace: list[dict] = []
    timings: dict = {}
    elapsed_seconds: float = 0.0
    scenario_match: Optional[dict] = None
    warnings: list[str] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    engine: dict = {}
    langsmith: dict = {}
    web_search: dict = {}
    routing: dict = {}

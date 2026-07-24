"""FastAPI 라우터.

책임: 입력 검증 + LangGraph invoke + 응답 성형. **라우팅 로직은 두지 않는다**
(모든 라우팅은 그래프 노드가 결정). 예외는 main.py 의 핸들러가 HTTP 코드로 매핑.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .dependencies import AppContext
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ResetRequest,
    ScenarioBlock,
    WarmupRequest,
)
from ..observability.langsmith import build_invoke_config

router = APIRouter()

_RUNID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_FNAME_RE = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


class InvalidRequestError(Exception):
    """message/action 형식 오류 → 400."""


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    ctx = _ctx(request)
    return HealthResponse(
        status="ok",
        engine=ctx.rag_adapter.status_dict(),
        langsmith={
            "tracing_enabled": bool(request.app.state.langsmith.get("tracing_enabled")),
            "project": request.app.state.langsmith.get("project"),
        },
        web_search={
            "enabled": ctx.settings.web_search_enabled,
            "scope": ctx.settings.web_search_scope,
            "provider": getattr(ctx.web_provider, "name", "unknown"),
        },
        routing={
            "backend": ctx.settings.rag3x_backend,
            "match_threshold": ctx.settings.scenario_match_threshold,
            "match_margin": ctx.settings.scenario_match_margin,
        },
    )


@router.get("/api/scenarios/root")
def scenarios_root(request: Request) -> dict:
    return _ctx(request).tree.root_payload()


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    ctx = _ctx(request)

    has_message = bool(body.message and body.message.strip())
    has_action = body.action is not None
    if has_message == has_action:
        # 둘 다 있거나 둘 다 없음
        raise InvalidRequestError("message 또는 action 중 정확히 하나가 필요합니다.")

    session_id = body.session_id or ctx.session_registry.new_session_id()
    thread_id = ctx.session_registry.thread_id(session_id)

    if has_action:
        act = body.action
        init_state = {
            "session_id": session_id,
            "thread_id": thread_id,
            "input_type": "action",
            "action_type": act.type,
            "action_scenario_id": act.scenario_id,
            "action_node_id": act.node_id,
            "selected_option_id": act.option_id,
            "action_label": act.label,
            "user_input": act.label,
        }
    else:
        init_state = {
            "session_id": session_id,
            "thread_id": thread_id,
            "input_type": "text",
            "user_input": body.message,
        }

    config = build_invoke_config(ctx.settings, session_id, thread_id)
    result = ctx.graph.invoke(init_state, config)

    timings = result.get("timings") or {}
    return ChatResponse(
        session_id=session_id,
        route=result.get("route"),
        answer=result.get("final_answer"),
        options=result.get("options") or [],
        scenario=ScenarioBlock(
            scenario_id=result.get("scenario_id"),
            node_id=result.get("current_node_id"),
            completed=bool(result.get("scenario_completed")),
        ),
        confidence=result.get("confidence"),
        answer_path=result.get("answer_path"),
        answer_source=result.get("answer_source"),
        evidence=result.get("evidence") or [],
        verification=result.get("verification"),
        source_meta=result.get("source_meta"),
        trace=result.get("trace") or [],
        timings=timings,
        elapsed_seconds=round(float(timings.get("total_s") or 0.0), 3),
        scenario_match=result.get("scenario_match"),
        warnings=result.get("warnings") or [],
    )


@router.post("/api/reset")
def reset(request: Request, body: ResetRequest) -> dict:
    ctx = _ctx(request)
    old_thread = ctx.session_registry.thread_id(body.session_id)
    new_thread = ctx.session_registry.reset(body.session_id)
    # best-effort: 이전 스레드 체크포인트 삭제
    try:
        ctx.checkpointer.delete_thread(old_thread)
    except Exception:
        pass
    return {"session_id": body.session_id, "reset": True, "thread_id": new_thread}


@router.post("/api/warmup")
def warmup(request: Request, body: WarmupRequest) -> dict:
    ctx = _ctx(request)
    deep = ctx.settings.rag3x_deep_warmup if body.deep is None else bool(body.deep)
    ctx.rag_adapter.start_warmup_background(deep=deep)
    return {"started": True, **ctx.rag_adapter.status_dict()}


@router.get("/evidence/{run_id}/{filename}")
def evidence(request: Request, run_id: str, filename: str):
    ctx = _ctx(request)
    if not _RUNID_RE.match(run_id) or not _FNAME_RE.match(filename):
        raise HTTPException(status_code=404, detail="not found")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=404, detail="not found")
    root = Path(ctx.settings.evidence_root).resolve()
    target = (root / run_id / filename).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=404, detail="not found")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target))

"""FastAPI 라우터 (v2).

책임: 입력 검증 + LangGraph invoke + 응답 성형. **라우팅 로직은 두지 않는다**
(모든 라우팅은 그래프 노드가 결정). 예외는 main.py 의 핸들러가 HTTP 코드로 매핑.

Phase 1: v1 동등(+run_id). Phase 3(clarify)·5(stream/feedback) 에서 확장.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .dependencies import AppContext
from .schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HealthResponse,
    ResetRequest,
    ScenarioBlock,
    WarmupRequest,
)
from ..observability.langsmith import build_invoke_config
from ..rag.adapter_util import RagBusyError, RagUnavailableError

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
    s = ctx.settings
    return HealthResponse(
        status="ok",
        engine=ctx.rag_adapter.status_dict(),
        langsmith={
            "tracing_enabled": bool(request.app.state.langsmith.get("tracing_enabled")),
            "project": request.app.state.langsmith.get("project"),
        },
        web_search={
            "enabled": s.web_search_enabled,
            "scope": s.web_search_scope,
            "provider": getattr(ctx.web_provider, "name", "unknown"),
            "model": getattr(ctx.web_provider, "model", None),
            # 어느 키를 쓰는지(값이 아니라 환경변수 이름만) — 유료 키 분리 확인용
            "key_source": getattr(ctx.web_provider, "key_source", None),
            "dedicated_key": s.web_search_api_key_present,
            # 오늘 호출/검색 수와 추정비용(provider 가 제공할 때만)
            "usage": (ctx.web_provider.usage()
                      if hasattr(ctx.web_provider, "usage") else None),
        },
        routing={
            "backend": s.rag_backend,
            "match_threshold": s.scenario_match_threshold,
            "match_margin": s.scenario_match_margin,
            "clarify_min_score": s.clarify_min_score,
        },
        toggles={
            "clarify": s.clarify_enabled,
            "composer_rag": s.composer_rag_enabled,
            "composer_faq": s.composer_faq_enabled,
            "contextualize": s.contextualize_enabled,
            "grader": s.grader_enabled,
            "rag_cache_ttl_s": s.rag_cache_ttl_s,
        },
        graph_mermaid=getattr(request.app.state, "graph_mermaid", None),
    )


@router.get("/api/scenarios/root")
def scenarios_root(request: Request) -> dict:
    return _ctx(request).tree.root_payload()


def _build_init_state(body: ChatRequest, session_id: str, thread_id: str) -> dict:
    has_message = bool(body.message and body.message.strip())
    has_action = body.action is not None
    has_clarify = body.clarify_response is not None
    if sum([has_message, has_action, has_clarify]) != 1:
        raise InvalidRequestError("message · action · clarify_response 중 정확히 하나가 필요합니다.")

    if has_action:
        act = body.action
        return {
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
    return {
        "session_id": session_id,
        "thread_id": thread_id,
        "input_type": "text",
        "user_input": body.message,
    }


def _shape_response(session_id: str, run_id: str, result: dict) -> ChatResponse:
    timings = result.get("timings") or {}
    return ChatResponse(
        session_id=session_id,
        type="answer",
        run_id=run_id,
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
        faq_evidence=result.get("faq_evidence") or [],
        verification=result.get("verification"),
        source_meta=result.get("source_meta"),
        trace=result.get("trace") or [],
        timings=timings,
        elapsed_seconds=round(float(timings.get("total_s") or 0.0), 3),
        scenario_match=result.get("scenario_match"),
        warnings=result.get("warnings") or [],
        original_answer=result.get("original_answer"),
        composed=bool(result.get("composed")),
        grader_verdict=result.get("grader_verdict"),
        citations=result.get("citations") or [],
    )


def _extract_interrupt(result: dict) -> dict | None:
    """graph.invoke 결과에 인터럽트(clarify 대기)가 있으면 그 payload 를 반환."""
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", None) or (first if isinstance(first, dict) else None)


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: Request, body: ChatRequest) -> ChatResponse:
    ctx = _ctx(request)

    session_id = body.session_id or ctx.session_registry.new_session_id()
    thread_id = ctx.session_registry.thread_id(session_id)
    run_id = str(uuid.uuid4())
    config = build_invoke_config(ctx.settings, session_id, thread_id, run_id=run_id)

    if body.clarify_response is not None:
        # clarify 되묻기 재개(HITL) — 같은 thread 로 resume.
        from langgraph.types import Command

        result = ctx.graph.invoke(
            Command(resume={"choice": body.clarify_response.choice}), config
        )
    else:
        init_state = _build_init_state(body, session_id, thread_id)
        result = ctx.graph.invoke(init_state, config)

    payload = _extract_interrupt(result)
    if payload is not None:
        return ChatResponse(
            session_id=session_id,
            type="clarify",
            run_id=run_id,
            route="clarify",
            answer=None,
            clarify={"candidates": payload.get("candidates") or []},
            trace=result.get("trace") or [],
        )
    return _shape_response(session_id, run_id, result)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/chat/stream")
def chat_stream(request: Request, body: ChatRequest):
    """SSE 스트리밍 대화.

    이벤트:
      - `progress` : {"stage","msg"}  — 노드가 get_stream_writer 로 발신한 진행상황
      - `node`     : {"node"}         — 방금 완료된 그래프 노드(파이프라인 시각화용)
      - `clarify`  : {"candidates"}   — HITL 되묻기로 일시정지
      - `final`    : ChatResponse 전체
      - `error`    : {"detail","status"}

    노드가 동기 함수라 동기 제너레이터를 쓴다(StreamingResponse 가 threadpool 에서 소비).
    """
    ctx = _ctx(request)
    session_id = body.session_id or ctx.session_registry.new_session_id()
    thread_id = ctx.session_registry.thread_id(session_id)
    run_id = str(uuid.uuid4())
    config = build_invoke_config(ctx.settings, session_id, thread_id, run_id=run_id)

    if body.clarify_response is not None:
        from langgraph.types import Command

        payload_in = Command(resume={"choice": body.clarify_response.choice})
    else:
        payload_in = _build_init_state(body, session_id, thread_id)   # 검증 포함(400)

    def gen():
        interrupt_payload = None
        try:
            yield _sse("progress", {"stage": "start", "msg": "요청을 처리하고 있어요…"})
            for mode, chunk in ctx.graph.stream(
                payload_in, config, stream_mode=["updates", "custom"]
            ):
                if mode == "custom":
                    yield _sse("progress", chunk if isinstance(chunk, dict) else {"msg": str(chunk)})
                elif mode == "updates" and isinstance(chunk, dict):
                    if "__interrupt__" in chunk:
                        intr = chunk["__interrupt__"]
                        first = intr[0] if intr else None
                        interrupt_payload = getattr(first, "value", None)
                        continue
                    for node_name in chunk:
                        yield _sse("node", {"node": node_name})

            if interrupt_payload is not None:
                yield _sse("clarify", {
                    "session_id": session_id,
                    "run_id": run_id,
                    "candidates": interrupt_payload.get("candidates") or [],
                })
                return

            result = ctx.graph.get_state(config).values
            yield _sse("final", _shape_response(session_id, run_id, result).model_dump())
        except (RagBusyError, RagUnavailableError) as exc:
            status = 429 if isinstance(exc, RagBusyError) else 503
            detail = ("이미 다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요."
                      if status == 429 else "RAG 엔진을 사용할 수 없습니다.")
            yield _sse("error", {"detail": detail, "status": status})
        except Exception:  # noqa: BLE001 - 내부 정보 노출 금지
            yield _sse("error", {"detail": "처리 중 오류가 발생했습니다.", "status": 500})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/api/feedback")
def feedback(request: Request, body: FeedbackRequest) -> dict:
    """👍/👎 를 LangSmith 피드백으로 기록. 추적 비활성이면 no-op(recorded=false)."""
    from ..observability.langsmith import send_feedback

    ok = send_feedback(body.run_id, body.score, body.comment)
    return {"recorded": bool(ok)}


@router.post("/api/reset")
def reset(request: Request, body: ResetRequest) -> dict:
    ctx = _ctx(request)
    old_thread = ctx.session_registry.thread_id(body.session_id)
    new_thread = ctx.session_registry.reset(body.session_id)
    try:
        ctx.checkpointer.delete_thread(old_thread)
    except Exception:
        pass
    return {"session_id": body.session_id, "reset": True, "thread_id": new_thread}


@router.post("/api/warmup")
def warmup(request: Request, body: WarmupRequest) -> dict:
    ctx = _ctx(request)
    deep = ctx.settings.rag_deep_warmup if body.deep is None else bool(body.deep)
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

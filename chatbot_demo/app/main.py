"""FastAPI 앱 팩토리.

- lifespan 에서 LangSmith 설정만 하고 rag3x 엔진은 만들지 않는다(지연 초기화).
- 예외 → HTTP 매핑(400/429/503/500). 응답에 API 키/절대경로를 넣지 않는다.
- static HTML 제공, evidence 라우트 포함.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import InvalidRequestError, router
from .dependencies import AppContext, build_context
from ..observability.langsmith import configure_langsmith
from ..rag.rag3x_adapter import RagBusyError, RagUnavailableError
from ..scenario.tree import InvalidActionError
from .graph.nodes import EmptyInputError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot_demo")


def create_app(ctx: Optional[AppContext] = None) -> FastAPI:
    if ctx is None:
        ctx = build_context()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.langsmith = configure_langsmith(ctx.settings)
        logger.info(
            "chatbot_demo 시작 (backend=%s, web_search=%s, tracing=%s)",
            ctx.settings.rag3x_backend,
            ctx.settings.web_search_enabled,
            app.state.langsmith.get("tracing_enabled"),
        )
        yield

    app = FastAPI(title="school-network-chatbot-demo", lifespan=lifespan)
    app.state.ctx = ctx
    app.state.langsmith = {"tracing_enabled": False, "project": ctx.settings.langsmith_project}

    # --- 예외 → HTTP 매핑 (키/절대경로 미포함) ---
    @app.exception_handler(EmptyInputError)
    async def _empty(request: Request, exc: EmptyInputError):
        return JSONResponse(status_code=400, content={"detail": "질문이 비어 있습니다."})

    @app.exception_handler(InvalidRequestError)
    async def _invalid_req(request: Request, exc: InvalidRequestError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(InvalidActionError)
    async def _invalid_action(request: Request, exc: InvalidActionError):
        return JSONResponse(status_code=400, content={"detail": "잘못된 시나리오 선택입니다."})

    @app.exception_handler(RagBusyError)
    async def _busy(request: Request, exc: RagBusyError):
        return JSONResponse(
            status_code=429,
            content={"detail": "이미 다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요."},
        )

    @app.exception_handler(RagUnavailableError)
    async def _unavail(request: Request, exc: RagUnavailableError):
        return JSONResponse(
            status_code=503,
            content={"detail": "RAG 엔진을 사용할 수 없습니다. 관리자에게 문의하세요."},
        )

    app.include_router(router)

    # --- static ---
    static_dir = Path(ctx.settings.static_dir)
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index():
        idx = static_dir / "index.html"
        if idx.is_file():
            return FileResponse(str(idx))
        return JSONResponse({"detail": "index.html 없음"}, status_code=404)

    return app

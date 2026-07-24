"""LangSmith 추적 연동.

- LANGSMITH_TRACING=true + LANGSMITH_API_KEY 존재 시 LangGraph 실행이 자동 추적된다
  (노드가 child run 으로 표시). 별도 @traceable 불필요.
- 키가 없으면 경고만 출력하고 tracing 을 강제 비활성화(백그라운드 업로드 오류 방지).
- API 키 값·절대경로·환경변수 전체는 절대 로깅하지 않는다.
"""

from __future__ import annotations

import logging
import os

from ..config.settings import Settings

logger = logging.getLogger("chatbot_demo.observability")

# 정적 태그
_BASE_TAGS = ["chatbot_demo", "langgraph", "scenario", "rag3x"]


def configure_langsmith(settings: Settings) -> dict:
    """프로세스 환경에 LangSmith 추적 설정을 반영한다.

    반환: {"tracing_enabled": bool, "project": str} (키 값은 포함하지 않음).
    """
    project = settings.langsmith_project
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)

    key_present = bool(os.environ.get("LANGSMITH_API_KEY")) or settings.langsmith_api_key_present

    if settings.langsmith_tracing and not key_present:
        logger.warning(
            "LangSmith tracing 이 요청되었으나 LANGSMITH_API_KEY 가 설정되지 않았습니다. "
            "tracing 을 비활성화합니다."
        )
        os.environ["LANGSMITH_TRACING"] = "false"
        return {"tracing_enabled": False, "project": project}

    enabled = bool(settings.langsmith_tracing and key_present)
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        logger.info("LangSmith tracing 활성화 (project=%s)", project)
    else:
        logger.info("LangSmith tracing 비활성화")
    return {"tracing_enabled": enabled, "project": project}


def build_invoke_config(
    settings: Settings, session_id: str, thread_id: str
) -> dict:
    """graph.invoke 용 config. thread_id + LangSmith tags/metadata 포함."""
    tags = list(_BASE_TAGS)
    tags.append(
        "web_search_enabled" if settings.web_search_enabled else "web_search_disabled"
    )
    return {
        "configurable": {"thread_id": thread_id},
        "tags": tags,
        "metadata": {"session_id": session_id},
        "run_name": "chat_turn",
    }


# LangSmith 에 절대 넣지 않을 키(방어적 필터)
_FORBIDDEN_META = {"api_key", "password", "secret", "token", "authorization"}


def attach_run_metadata(meta: dict, *, tags: list | None = None) -> None:
    """최상위(turn) run 에 동적 metadata(+tags)를 붙인다(가능할 때만).

    tracing 이 꺼져 있거나 langsmith 가 없으면 조용히 no-op.
    비밀스러운 키는 필터링한다.
    """
    _apply_to_run(meta, tags=tags, to_root=True)


def _filter_safe(meta: dict) -> dict:
    return {
        k: v
        for k, v in meta.items()
        if k.lower() not in _FORBIDDEN_META and v is not None
    }


def _apply_to_run(meta: dict, *, tags: list | None = None, to_root: bool = False) -> None:
    """현재(노드) run 또는 최상위 run 에 metadata/tags 를 붙인다. 실패 시 no-op."""
    safe = _filter_safe(meta or {})
    if not safe and not tags:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return
        if to_root:
            # 부모 체인을 따라 최상위(그래프 turn) run 으로 이동
            guard = 0
            while getattr(run, "parent_run", None) is not None and guard < 20:
                run = run.parent_run
                guard += 1
        if run.extra is None:
            run.extra = {}
        if safe:
            md = run.extra.setdefault("metadata", {})
            md.update(safe)
        if tags:
            existing = list(getattr(run, "tags", None) or [])
            for t in tags:
                if t not in existing:
                    existing.append(t)
            run.tags = existing
    except Exception:
        # 추적 비활성/미설치 등: 무시
        return


def add_node_metadata(meta: dict, *, tags: list | None = None) -> None:
    """현재 노드 run 에 판단 근거 metadata(+tags)를 붙인다. 실패 시 no-op."""
    _apply_to_run(meta, tags=tags, to_root=False)


def traced_call(name: str, fn, run_type: str = "tool", **kwargs):
    """fn 을 name 이름의 traceable child run 으로 실행한다.

    langsmith 미설치/추적 비활성이면 그냥 fn 을 호출한다(무해).
    fn 의 반환값이 run 의 outputs 로 기록된다.
    fn 자체의 예외는 삼키지 않고 정상 전파한다(중복 실행 방지).
    """
    try:
        from langsmith import traceable

        wrapped = traceable(name=name, run_type=run_type)(fn)
    except Exception:
        # langsmith 미설치/데코레이터 생성 실패: 추적 없이 그대로 실행
        return fn(**kwargs)
    return wrapped(**kwargs)

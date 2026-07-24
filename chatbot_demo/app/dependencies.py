"""애플리케이션 컨텍스트(의존성 주입 컨테이너)와 세션 레지스트리."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver

from ..config.settings import Settings, load_settings
from ..scenario.loader import load_faq, load_scenarios
from ..scenario.matcher import ScenarioMatcher
from ..scenario.models import FaqStore
from ..scenario.tree import ScenarioTree
from ..web_search.disabled import DisabledWebSearchProvider
from .graph.builder import build_graph


class SessionRegistry:
    """session_id → epoch 매핑. reset 시 epoch 를 올려 새 체크포인트 스레드를 쓴다."""

    def __init__(self):
        self._epochs: dict[str, int] = {}
        self._lock = threading.Lock()

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def thread_id(self, session_id: str) -> str:
        with self._lock:
            epoch = self._epochs.get(session_id, 0)
        return f"{session_id}:{epoch}"

    def reset(self, session_id: str) -> str:
        with self._lock:
            self._epochs[session_id] = self._epochs.get(session_id, 0) + 1
        return self.thread_id(session_id)


@dataclass
class AppContext:
    settings: Settings
    faq: FaqStore
    tree: ScenarioTree
    matcher: ScenarioMatcher
    rag_adapter: Any
    web_provider: Any
    checkpointer: Any
    graph: Any
    session_registry: SessionRegistry


def build_context(
    settings: Optional[Settings] = None,
    *,
    rag_adapter: Any = None,
    web_provider: Any = None,
    checkpointer: Any = None,
) -> AppContext:
    """AppContext 를 조립한다. rag_adapter/web_provider 를 주입하면(테스트) 대체된다."""
    settings = settings or load_settings()

    faq = load_faq(settings.faq_path)
    tree = load_scenarios(settings.scenarios_path, faq)
    matcher = ScenarioMatcher(
        faq,
        threshold=settings.scenario_match_threshold,
        margin=settings.scenario_match_margin,
    )

    if rag_adapter is None:
        # 실제 어댑터(지연 초기화 — 여기서 엔진을 만들지 않음)
        from ..rag.rag3x_adapter import Rag3xAdapter

        rag_adapter = Rag3xAdapter(settings)

    if web_provider is None:
        web_provider = DisabledWebSearchProvider()

    if checkpointer is None:
        checkpointer = InMemorySaver()

    ctx = AppContext(
        settings=settings,
        faq=faq,
        tree=tree,
        matcher=matcher,
        rag_adapter=rag_adapter,
        web_provider=web_provider,
        checkpointer=checkpointer,
        graph=None,
        session_registry=SessionRegistry(),
    )
    ctx.graph = build_graph(ctx, checkpointer=checkpointer)
    return ctx

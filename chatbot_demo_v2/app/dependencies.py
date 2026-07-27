"""애플리케이션 컨텍스트(의존성 주입 컨테이너)와 세션 레지스트리 (v2).

Phase 1: v1 동등. rag_adapter(blackbox) 주입 가능.
Phase 2 에서 rag_config/rag_backend/rag_lock/current_metrics 팩토리와 RAG 서브그래프 배선 추가.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from langgraph.checkpoint.memory import InMemorySaver

from ..config.settings import Settings, load_settings
from ..scenario.loader import load_faq, load_scenarios
from ..scenario.matcher import ScenarioMatcher, SemanticScenarioMatcher
from ..scenario.models import FaqStore
from ..scenario.tree import ScenarioTree
from ..web_search.disabled import DisabledWebSearchProvider
from ..graph.builder import build_graph


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
    llm: Any = None          # LlmHelper (contextualize/composer/grader) — 지연 초기화
    prompts: Any = None      # PromptLoader (prompts/*.md 핫리로드)
    faq_links: dict = None   # faq_id → [{doc_slug, document_name, pages}] (근거 이미지용)


def build_context(
    settings: Optional[Settings] = None,
    *,
    rag_adapter: Any = None,
    web_provider: Any = None,
    checkpointer: Any = None,
    llm: Any = None,
) -> AppContext:
    """AppContext 를 조립한다. rag_adapter/web_provider/llm 을 주입하면(테스트) 대체된다."""
    settings = settings or load_settings()

    faq = load_faq(settings.faq_path)
    tree = load_scenarios(settings.scenarios_path, faq)
    # 2026-07-27: 기본은 의미 매칭(임베딩+리랭커). 임베딩 파일/백엔드가 없으면
    # SemanticScenarioMatcher 내부에서 문자 유사도로 조용히 폴백한다.
    if settings.scenario_match_backend == "semantic":
        matcher = SemanticScenarioMatcher(
            faq,
            threshold=settings.scenario_match_threshold,
            margin=settings.scenario_match_margin,
            settings=settings,
        )
    else:
        matcher = ScenarioMatcher(
            faq,
            threshold=settings.scenario_match_threshold,
            margin=settings.scenario_match_margin,
        )

    if rag_adapter is None:
        # 프로덕션: RAG 서브그래프 어댑터(지연 초기화 — 여기서 엔진을 만들지 않음).
        # Rag3xAdapter(블랙박스)도 adapter_util 에 남겨 둠(비상 롤백용).
        from ..rag.adapter_util import SubgraphRagAdapter

        rag_adapter = SubgraphRagAdapter(settings)

    if web_provider is None:
        web_provider = DisabledWebSearchProvider()

    if checkpointer is None:
        checkpointer = InMemorySaver()

    if llm is None:
        from ..rag.llm_helper import LlmHelper

        llm = LlmHelper(settings)

    from ..prompts.loader import PromptLoader

    prompts = PromptLoader(settings.prompts_dir)

    # FAQ 근거 링크(없으면 빈 dict — 근거 이미지만 생략되고 답변은 정상)
    faq_links: dict = {}
    try:
        import json

        p = settings.faq_doc_links_path
        if p.is_file():
            faq_links = json.loads(p.read_text(encoding="utf-8")).get("links", {})
    except Exception:  # noqa: BLE001
        faq_links = {}

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
        llm=llm,
        prompts=prompts,
        faq_links=faq_links,
    )
    ctx.graph = build_graph(ctx, checkpointer=checkpointer)
    return ctx

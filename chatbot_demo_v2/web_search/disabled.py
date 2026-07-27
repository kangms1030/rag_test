"""비활성 웹검색 provider (기본값). 절대 외부 호출을 하지 않는다."""

from __future__ import annotations

from .base import web_result


class DisabledWebSearchProvider:
    name = "disabled"

    def search_and_answer(self, question: str, context: dict | None = None) -> dict:
        # 어떤 경우에도 네트워크 호출 없음.
        return web_result(
            answer="",
            provider=self.name,
            enabled=False,
            note="웹검색이 비활성화되어 있습니다 (WEB_SEARCH_ENABLED=false).",
        )

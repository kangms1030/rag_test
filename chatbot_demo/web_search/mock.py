"""테스트/시연용 Mock 웹검색 provider. 실제 네트워크 호출은 하지 않는다."""

from __future__ import annotations

from .base import web_result


class MockWebSearchProvider:
    name = "mock"

    def __init__(self, canned_answer: str | None = None):
        self.call_count = 0
        self.last_question: str | None = None
        self._canned = canned_answer or "(mock) 웹검색 결과 기반 예시 답변입니다."

    def search_and_answer(self, question: str, context: dict | None = None) -> dict:
        self.call_count += 1
        self.last_question = question
        return web_result(
            answer=self._canned,
            provider=self.name,
            enabled=True,
            sources=[{"title": "예시 출처", "url": "https://example.org/mock"}],
            note="Mock provider — 실제 검색이 아닙니다.",
        )

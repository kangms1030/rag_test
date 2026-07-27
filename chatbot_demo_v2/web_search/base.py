"""웹검색 provider 인터페이스.

실제 외부 LLM/검색 provider는 이번 데모에서 특정하지 않는다.
기본은 Disabled(호출 안 함)이며, 테스트/시연용 Mock을 제공한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


def web_result(
    *,
    answer: str,
    provider: str,
    enabled: bool,
    sources: list[dict] | None = None,
    note: str | None = None,
) -> dict:
    """표준 web_result dict 생성기."""
    return {
        "answer": answer,
        "provider": provider,
        "enabled": enabled,
        "sources": sources or [],
        "note": note,
    }


@runtime_checkable
class WebSearchProvider(Protocol):
    name: str

    def search_and_answer(self, question: str, context: dict | None = None) -> dict:
        """질문에 대해 웹검색 기반 답변을 반환한다.

        반환: web_result() 형태의 dict.
        Disabled provider는 실제 네트워크 호출 없이 비활성 표시만 반환한다.
        """
        ...

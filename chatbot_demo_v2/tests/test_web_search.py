"""웹검색(Gemini Grounding) provider · 도메인 게이트 · 그래프 배선 테스트.

실제 네트워크 호출은 하지 않는다 — HTTP poster 를 주입해 응답을 흉내 낸다.
"""

from __future__ import annotations

import pytest

from chatbot_demo_v2.app.dependencies import build_context, build_web_provider
from chatbot_demo_v2.config.settings import load_settings
from chatbot_demo_v2.graph.routing import parse_domain_verdict
from chatbot_demo_v2.rag.adapter_util import FakeRagAdapter
from chatbot_demo_v2.web_search.base import web_result
from chatbot_demo_v2.web_search.disabled import DisabledWebSearchProvider
from chatbot_demo_v2.web_search.gemini_grounding import GeminiGroundingProvider
from chatbot_demo_v2.web_search.mock import MockWebSearchProvider


# ---------- 픽스처/헬퍼 ----------
GROUNDED_RESPONSE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "무선 AP 설치는 관할 교육청에 신청해야 합니다."}]},
            "groundingMetadata": {
                "webSearchQueries": ["학교 무선 AP 설치 신청 절차"],
                "groundingChunks": [
                    {"web": {"uri": "https://example.org/a", "title": "A 문서"}},
                    {"web": {"uri": "https://example.org/a", "title": "A 문서(중복)"}},
                    {"web": {"uri": "https://example.org/b", "title": "B 문서"}},
                ],
                "searchEntryPoint": {"renderedContent": "<div>검색 추천</div>"},
            },
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
        "thoughtsTokenCount": 10,
    },
}


class _Resp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class _Poster:
    """requests.post 대역. 호출 인자를 기록하고 정해진 응답을 돌려준다."""

    def __init__(self, *responses: _Resp):
        self.calls: list[dict] = []
        self._responses = list(responses)

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    """실제 키가 섞이지 않게 공용 키는 가짜로, 전용 키는 없는 상태로 고정한다."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.delenv("WEB_SEARCH_GEMINI_API_KEY", raising=False)


def _provider(poster, **kw) -> GeminiGroundingProvider:
    kw.setdefault("max_retries", 0)
    return GeminiGroundingProvider(poster=poster, **kw)


class StubLlm:
    """도메인 게이트용 LLM 대역."""

    def __init__(self, verdict: str | None):
        self.verdict = verdict
        self.calls = 0

    def chat(self, prompt: str):
        self.calls += 1
        return self.verdict


def _text(msg, sid="s", tid="s:0"):
    return {"session_id": sid, "thread_id": tid, "input_type": "text", "user_input": msg}


def _abstain_rag():
    return FakeRagAdapter(result={
        "run_id": "r", "final_answer": "", "answer_path": "none",
        "confidence": "unknown", "verification": None,
        "evidence": [], "metrics": {}, "selected_pages": [],
    })


def _web_env(tmp_path, **extra) -> dict:
    env = {
        "WEB_SEARCH_ENABLED": "true",
        "WEB_SEARCH_SCOPE": "in_domain_unresolved",
        "DEMO_EVIDENCE_DIR": str(tmp_path / "ev"),
        # 이 파일의 관심사는 '웹검색 게이트'다 — 앞단(FAQ 매칭·되묻기)이 먼저 가로채지
        # 않도록 임계를 올리고 clarify 를 끈 채 RAG 경로로 흘려보낸다.
        "SCENARIO_MATCH_THRESHOLD": "0.99",
        "CLARIFY_ENABLED": "false",
    }
    env.update(extra)
    return env


# ---------- provider ----------
def test_provider_builds_grounding_payload():
    poster = _Poster(_Resp(200, GROUNDED_RESPONSE))
    res = _provider(poster).search_and_answer("학교에 AP 설치하려면?", context={"system": "지시문"})

    body = poster.calls[0]["json"]
    assert body["tools"] == [{"google_search": {}}]          # 검색 grounding 활성
    assert body["systemInstruction"]["parts"][0]["text"] == "지시문"
    assert body["contents"][0]["parts"][0]["text"] == "학교에 AP 설치하려면?"
    assert "generateContent" in poster.calls[0]["url"]
    assert res["enabled"] is True
    assert res["answer"].startswith("무선 AP")


def test_provider_parses_sources_and_usage():
    res = _provider(_Poster(_Resp(200, GROUNDED_RESPONSE))).search_and_answer("질문")

    assert [s["url"] for s in res["sources"]] == [
        "https://example.org/a", "https://example.org/b",       # 중복 URL 제거
    ]
    assert res["search_queries"] == ["학교 무선 AP 설치 신청 절차"]
    assert res["search_entry_point"] == "<div>검색 추천</div>"
    assert res["usage"]["searches"] == 1
    assert res["usage"]["thought_tokens"] == 10
    assert res["usage"]["est_cost_usd"] > 0
    assert res["model"] == "gemini-3.1-flash-lite"


def test_provider_max_sources_limit():
    res = _provider(_Poster(_Resp(200, GROUNDED_RESPONSE)), max_sources=1).search_and_answer("q")
    assert len(res["sources"]) == 1


def test_provider_discards_answer_without_sources():
    """검색을 타지 않은(=근거 없는) 답변은 채택하지 않는다."""
    payload = {"candidates": [{"content": {"parts": [{"text": "모델 내부지식 답변"}]}}]}
    res = _provider(_Poster(_Resp(200, payload))).search_and_answer("q")
    assert res["answer"] == ""
    assert "근거" in res["note"]


def test_provider_http_error_returns_empty_not_raises():
    poster = _Poster(_Resp(403, {"error": {"message": "billing required"}}))
    res = _provider(poster).search_and_answer("q")
    assert res["answer"] == ""
    assert "403" in res["note"]
    assert len(poster.calls) == 1          # 4xx 는 재시도하지 않는다


def test_provider_network_error_returns_empty_not_raises():
    def boom(url, **kwargs):
        raise TimeoutError("timeout")

    res = _provider(boom).search_and_answer("q")
    assert res["answer"] == ""
    assert "실패" in res["note"]


def test_provider_daily_budget_blocks_second_call():
    poster = _Poster(_Resp(200, GROUNDED_RESPONSE))
    p = _provider(poster, daily_budget=1)
    assert p.search_and_answer("q1")["answer"]
    blocked = p.search_and_answer("q2")
    assert blocked["answer"] == ""
    assert "한도" in blocked["note"]
    assert len(poster.calls) == 1
    assert p.usage()["calls_today"] == 1


def test_provider_never_logs_key(caplog):
    poster = _Poster(_Resp(403, {}))
    _provider(poster).search_and_answer("q")
    assert "test-key-not-real" not in caplog.text
    assert poster.calls[0]["params"]["key"] == "test-key-not-real"   # 키는 URL 파라미터로만


# ---------- 웹검색 전용 키 ----------
def test_dedicated_key_takes_precedence(monkeypatch):
    """웹검색 전용 키가 있으면 그 키로 나간다(유료 티어 분리)."""
    monkeypatch.setenv("WEB_SEARCH_GEMINI_API_KEY", "paid-key")
    poster = _Poster(_Resp(200, GROUNDED_RESPONSE))
    p = _provider(poster)
    p.search_and_answer("q")
    assert poster.calls[0]["params"]["key"] == "paid-key"
    assert p.key_source == "WEB_SEARCH_GEMINI_API_KEY"
    assert p.usage()["key_source"] == "WEB_SEARCH_GEMINI_API_KEY"


def test_falls_back_to_common_key(monkeypatch):
    """전용 키가 없으면 공용 GEMINI_API_KEY 로 폴백한다(기존 동작)."""
    monkeypatch.delenv("WEB_SEARCH_GEMINI_API_KEY", raising=False)
    poster = _Poster(_Resp(200, GROUNDED_RESPONSE))
    p = _provider(poster)
    p.search_and_answer("q")
    assert poster.calls[0]["params"]["key"] == "test-key-not-real"
    assert p.key_source == "GEMINI_API_KEY"


def test_no_key_at_all_raises(monkeypatch):
    monkeypatch.delenv("WEB_SEARCH_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _provider(_Poster(_Resp(200, {})))


def test_settings_reports_dedicated_key(tmp_path):
    s = load_settings(env=_web_env(tmp_path, WEB_SEARCH_GEMINI_API_KEY="paid-key"))
    assert s.web_search_api_key_present is True
    assert load_settings(env=_web_env(tmp_path)).web_search_api_key_present is False


# ---------- provider 팩토리 ----------
def test_factory_disabled_when_web_search_off(tmp_path):
    settings = load_settings(env={"WEB_SEARCH_ENABLED": "false"})
    assert isinstance(build_web_provider(settings), DisabledWebSearchProvider)


def test_factory_gemini_when_enabled(tmp_path):
    settings = load_settings(env=_web_env(tmp_path))
    assert isinstance(build_web_provider(settings), GeminiGroundingProvider)


def test_factory_mock_and_unknown(tmp_path):
    mock_s = load_settings(env=_web_env(tmp_path, WEB_SEARCH_PROVIDER="mock"))
    assert isinstance(build_web_provider(mock_s), MockWebSearchProvider)
    unknown_s = load_settings(env=_web_env(tmp_path, WEB_SEARCH_PROVIDER="bing"))
    assert isinstance(build_web_provider(unknown_s), DisabledWebSearchProvider)


def test_factory_falls_back_when_key_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = load_settings(env=_web_env(tmp_path))
    assert isinstance(build_web_provider(settings), DisabledWebSearchProvider)


# ---------- 도메인 게이트 ----------
def test_parse_domain_verdict():
    assert parse_domain_verdict("IN_DOMAIN") is True
    assert parse_domain_verdict("판정: OUT_OF_DOMAIN") is False   # 부분문자열 오탐 없음
    assert parse_domain_verdict("잘 모르겠습니다") is None
    assert parse_domain_verdict(None) is None


def test_gate_pass_calls_web(tmp_path):
    settings = load_settings(env=_web_env(tmp_path))
    web = MockWebSearchProvider(canned_answer="웹 답변")
    llm = StubLlm("IN_DOMAIN")
    ctx = build_context(settings, rag_adapter=_abstain_rag(), web_provider=web, llm=llm)
    r = ctx.graph.invoke(_text("학교 무선 AP 새로 설치하려면 어디에 신청하나요", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert web.call_count == 1
    assert r["route"] == "web_search"
    assert r["answer_source"] == "web"
    assert r["warnings"]                       # 웹 근거 주의 문구
    assert r["source_meta"]["type"] == "web"


def test_gate_block_out_of_domain_skips_web(tmp_path):
    """범위 밖 질문이면 유료 웹검색을 호출하지 않고 보류한다."""
    settings = load_settings(env=_web_env(tmp_path))
    web = MockWebSearchProvider()
    ctx = build_context(settings, rag_adapter=_abstain_rag(), web_provider=web,
                        llm=StubLlm("OUT_OF_DOMAIN"))
    r = ctx.graph.invoke(_text("오늘 저녁 메뉴 추천해줘", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert web.call_count == 0
    assert r["route"] == "abstain"
    assert r["confidence"] == "abstain"


def test_gate_unknown_verdict_skips_web(tmp_path):
    """LLM 미응답(판정 불가)이면 호출하지 않는다 — 확인되지 않은 지출을 만들지 않는다."""
    settings = load_settings(env=_web_env(tmp_path))
    web = MockWebSearchProvider()
    ctx = build_context(settings, rag_adapter=_abstain_rag(), web_provider=web,
                        llm=StubLlm(None))
    r = ctx.graph.invoke(_text("학교 무선랜 속도가 느립니다", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert web.call_count == 0
    assert r["route"] == "abstain"


def test_scope_any_unresolved_skips_gate(tmp_path):
    settings = load_settings(env=_web_env(tmp_path, WEB_SEARCH_SCOPE="any_unresolved"))
    web = MockWebSearchProvider(canned_answer="웹 답변")
    llm = StubLlm("OUT_OF_DOMAIN")     # 게이트를 건너뛰므로 호출되지 않아야 한다
    ctx = build_context(settings, rag_adapter=_abstain_rag(), web_provider=web, llm=llm)
    r = ctx.graph.invoke(_text("무엇이든 물어보세요 12345", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert llm.calls == 0
    assert web.call_count == 1
    assert r["route"] == "web_search"


class EmptyWebProvider:
    """웹검색이 답을 못 찾은 경우(답변 빈 문자열)."""

    name = "empty"

    def __init__(self):
        self.call_count = 0

    def search_and_answer(self, question: str, context: dict | None = None) -> dict:
        self.call_count += 1
        return web_result(answer="", provider=self.name, enabled=True,
                          note="웹 검색에서도 답변을 찾지 못했습니다.")


#: 실사용의 대표적인 '반려' 답변 — 빈 답이 아니라 "자료에 없다"는 답이다.
WEAK_ANSWER = "제공된 자료에서 확인할 수 없습니다."


class GraderLlm:
    """프롬프트별로 다르게 답하는 LLM 대역.

    게이트(web_domain_gate)와 해결도 판정(answer_grader) 프롬프트에만 판정어를 돌려주고,
    나머지(composer/contextualize)는 None → pass-through(원문 유지)로 둔다.
    두 프롬프트만 각각 IN_DOMAIN / UNRESOLVED 토큰을 포함한다(다른 프롬프트에는 없음).
    """

    def __init__(self, domain: str = "IN_DOMAIN", composed: str = WEAK_ANSWER):
        self.domain = domain
        self.composed = composed
        self.prompts: list[str] = []

    def chat(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        if "IN_DOMAIN" in prompt:
            return self.domain
        if "UNRESOLVED" in prompt:
            return "UNRESOLVED"
        # composer — grader 는 '합성 결과'를 판정하므로 합성이 없으면 판정 자체가 생략된다.
        return self.composed


def _weak_rag():
    """답변은 냈지만 '자료에서 확인할 수 없습니다' 류인 저신뢰 응답(실사용의 대표 반려)."""
    return FakeRagAdapter(result={
        "run_id": "r", "final_answer": WEAK_ANSWER,
        "answer_path": "text", "confidence": "low", "verification": None,
        "evidence": [], "metrics": {}, "selected_pages": [],
    })


def test_unresolved_rag_answer_escalates_to_web(tmp_path):
    """RAG 가 답을 내긴 했지만 grader 가 UNRESOLVED 로 보면 웹검색까지 간다(2026-08-03)."""
    settings = load_settings(env=_web_env(tmp_path))
    web = MockWebSearchProvider(canned_answer="웹에서 찾은 답변")
    ctx = build_context(settings, rag_adapter=_weak_rag(), web_provider=web, llm=GraderLlm())
    r = ctx.graph.invoke(_text("분당 지역 AP 공급사가 어디인가요", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert web.call_count == 1
    assert r["route"] == "web_search"
    assert r["final_answer"] == "웹에서 찾은 답변"
    assert r["answer_source"] == "web"
    # 버려진 RAG 초안에 대한 경고는 남기지 않는다
    assert all("웹 검색 결과" in w for w in r["warnings"])


def test_unresolved_rag_answer_keeps_answer_when_web_off(tmp_path):
    """웹검색이 꺼져 있으면 기존 동작 그대로 — RAG 답변 + 미해결 경고."""
    settings = load_settings(env=_web_env(tmp_path, WEB_SEARCH_ENABLED="false"))
    ctx = build_context(settings, rag_adapter=_weak_rag(), llm=GraderLlm())
    r = ctx.graph.invoke(_text("분당 지역 AP 공급사가 어디인가요", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert r["route"] == "rag3x"
    assert r["final_answer"] == WEAK_ANSWER
    assert r["confidence"] == "low"


def test_web_failure_keeps_prior_rag_answer(tmp_path):
    """웹검색이 빈손이면 직전 RAG 답변으로 되돌아간다(보류로 떨어뜨리지 않는다)."""
    settings = load_settings(env=_web_env(tmp_path))
    web = EmptyWebProvider()
    ctx = build_context(settings, rag_adapter=_weak_rag(), web_provider=web, llm=GraderLlm())
    r = ctx.graph.invoke(_text("분당 지역 AP 공급사가 어디인가요", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert web.call_count == 1
    assert r["route"] == "rag3x"
    assert r["final_answer"] == WEAK_ANSWER


def test_gate_block_keeps_prior_rag_answer(tmp_path):
    """범위 밖 판정이면 웹검색을 부르지 않고, 직전 답변이 있으면 그대로 제시한다."""
    settings = load_settings(env=_web_env(tmp_path))
    web = MockWebSearchProvider()
    ctx = build_context(settings, rag_adapter=_weak_rag(), web_provider=web,
                        llm=GraderLlm(domain="OUT_OF_DOMAIN"))
    r = ctx.graph.invoke(_text("저녁 메뉴 추천", "s", "t"), {"configurable": {"thread_id": "t"}})
    assert web.call_count == 0
    assert r["route"] == "rag3x"
    assert r["final_answer"] == WEAK_ANSWER


def test_empty_web_answer_falls_back_to_abstain(tmp_path):
    """웹검색까지 무응답이면 빈 말풍선 대신 보류 안내."""
    settings = load_settings(env=_web_env(tmp_path))
    web = EmptyWebProvider()
    ctx = build_context(settings, rag_adapter=_abstain_rag(), web_provider=web,
                        llm=StubLlm("IN_DOMAIN"))
    r = ctx.graph.invoke(_text("학교 스위치 포트가 안 켜집니다", "s", "t"),
                         {"configurable": {"thread_id": "t"}})
    assert web.call_count == 1
    assert r["confidence"] == "abstain"
    assert r["answer_source"] == "none"
    assert r["final_answer"]

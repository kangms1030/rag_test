"""Gemini Grounding(Google 검색) 웹검색 provider.

내부 자료(RAG)로 답하지 못한 **상담 범위 안** 질문에만 호출된다 —
범위 판정(도메인 게이트)은 graph/nodes.web_search_answer 가 담당하고, 이 모듈은
실제 호출·파싱·비용 계측만 한다.

- API: generativelanguage v1beta ``generateContent`` + ``tools:[{"google_search":{}}]``.
  requests 만 사용해 신규 의존성이 없다(rag3x/gemini_backend.py 와 같은 방식).
- 실패(키 없음/네트워크/429/일일예산 초과)해도 **예외를 밖으로 던지지 않는다** —
  answer="" 를 돌려주고 final_formatter 가 보류 안내로 처리한다.
  로컬 폴백은 두지 않는다(웹검색은 보완재이지 대체재가 아니다).
- 비용: Gemini 3.x 는 "모델이 실행한 검색 쿼리 수"로 과금된다(응답의 webSearchQueries).
  호출마다 검색 수·토큰을 기록하고, 일일 상한으로 과금 폭주를 막는다.
- 키(GEMINI_API_KEY)는 로그·예외 메시지·응답 어디에도 노출하지 않는다.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date

from .base import web_result

logger = logging.getLogger("chatbot_demo_v2.web_search")

_API_HOST = "https://generativelanguage.googleapis.com/v1beta/models"

# 참고용 추정 요율(2026-08 공개요율 기준, USD). 실제 청구액은 Google 콘솔이 기준이다.
#   gemini-3.1-flash-lite 유료 티어: 입력 $0.25 / 출력 $1.50 (1M 토큰)
#   Grounding with Google Search: 월 5,000회 무료(Gemini 3.x 공유) 후 $14 / 1,000회
_PRICE_IN_PER_M = 0.25
_PRICE_OUT_PER_M = 1.50
_PRICE_PER_SEARCH = 0.014

DEFAULT_SYSTEM_PROMPT = (
    "당신은 대한민국 학교 유·무선 네트워크(스쿨넷) 장애상담 안내자입니다. "
    "내부 자료에서 답을 찾지 못해 웹 검색으로 보완하는 상황입니다.\n"
    "- 반드시 검색 결과에 근거해 한국어 존댓말로 3~6문장으로 답합니다.\n"
    "- 결론(또는 가장 먼저 할 조치)을 첫 문장에 씁니다. 절차가 여러 단계면 번호 목록을 씁니다.\n"
    "- 검색으로 확인되지 않는 내용은 지어내지 말고, 확인이 필요한 범위를 밝힙니다.\n"
    "- 특정 학교의 내부 규정·계약·장비 현황처럼 웹에서 확인할 수 없는 정보는 추측하지 말고 "
    "학교 정보부 담당 선생님 또는 스쿨넷 서비스 지원센터(1899-0979) 문의를 안내합니다."
)


#: 웹검색 전용 키를 먼저 찾고, 없으면 공용 키로 폴백한다.
#: 검색 grounding 은 **유료 티어에서만** 동작하므로(무료 키는 429 RESOURCE_EXHAUSTED),
#: 결제를 뚫어 둔 키를 RAG 용 무료 키와 분리해서 쓸 수 있어야 한다.
KEY_ENV_NAMES = ("WEB_SEARCH_GEMINI_API_KEY", "GEMINI_API_KEY")


def _load_api_key() -> tuple[str, str]:
    """(키, 어느 환경변수에서 왔는지) 반환. **키 값은 절대 로깅/노출하지 않는다.**"""
    for name in KEY_ENV_NAMES:
        key = os.environ.get(name, "").strip()
        if key:
            return key, name
    raise RuntimeError(
        f"웹검색용 API 키 미설정 — {' 또는 '.join(KEY_ENV_NAMES)} 중 하나가 필요합니다."
    )


class GeminiGroundingProvider:
    """Gemini + Google 검색 grounding 기반 웹검색 provider."""

    name = "gemini_grounding"

    def __init__(
        self,
        *,
        model: str = "gemini-3.1-flash-lite",
        timeout_s: int = 30,
        max_retries: int = 2,
        max_sources: int = 5,
        daily_budget: int = 100,
        system_prompt: str | None = None,
        poster=None,
        api_key: str | None = None,
    ):
        import requests  # 기설치(gemini_backend 와 동일)

        self._post_fn = poster or requests.post   # 테스트에서 주입
        if api_key:
            self._key, self.key_source = api_key.strip(), "explicit"
        else:
            self._key, self.key_source = _load_api_key()
        logger.info("[web] 웹검색 키 출처=%s (값은 기록하지 않음)", self.key_source)
        self.model = model
        self._timeout = int(timeout_s)
        self._max_retries = int(max_retries)
        self._max_sources = int(max_sources)
        self._daily_budget = int(daily_budget)
        self._system = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._lock = threading.Lock()
        self._day = date.today()
        self.calls_today = 0
        self.searches_today = 0
        self.cost_today_usd = 0.0

    # ---------- 일일 예산 ----------
    def _take_budget(self) -> bool:
        """호출 1건을 예산에서 차감. 상한 초과면 False(호출하지 않음). 0 이하면 무제한."""
        with self._lock:
            today = date.today()
            if today != self._day:
                self._day = today
                self.calls_today = 0
                self.searches_today = 0
                self.cost_today_usd = 0.0
            if self._daily_budget > 0 and self.calls_today >= self._daily_budget:
                return False
            self.calls_today += 1
            return True

    def usage(self) -> dict:
        with self._lock:
            return {
                "key_source": self.key_source,      # 어느 env 키를 쓰는지(값은 미노출)
                "date": str(self._day),
                "calls_today": self.calls_today,
                "searches_today": self.searches_today,
                "daily_budget": self._daily_budget,
                "est_cost_today_usd": round(self.cost_today_usd, 6),
            }

    # ---------- 호출 ----------
    def _post(self, payload: dict) -> dict:
        """generateContent 1회 호출 + 429/5xx 지수 백오프. 키는 URL 파라미터, 로그 비노출."""
        url = f"{_API_HOST}/{self.model}:generateContent"
        last = "unknown"
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._post_fn(
                    url,
                    params={"key": self._key},
                    json=payload,
                    timeout=self._timeout,
                    headers={"Content-Type": "application/json"},
                )
            except Exception as e:  # noqa: BLE001 - 네트워크/타임아웃
                last = type(e).__name__
                logger.warning("[web] 네트워크 오류(%s) — 재시도 %d/%d", last, attempt + 1,
                               self._max_retries)
            else:
                if resp.status_code == 200:
                    return resp.json()
                last = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 500, 502, 503, 504):
                    # 4xx(키·권한·요금제 오류)는 재시도해도 같다. 본문은 키가 섞일 수 있어
                    # 상태코드만 표면화한다. 무료 티어 키는 grounding 이 막혀 여기로 떨어진다.
                    break
                logger.warning("[web] %s — 재시도 %d/%d", last, attempt + 1, self._max_retries)
            if attempt < self._max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(last)

    def _record(self, searches: int, usage: dict) -> float:
        """검색 수·토큰으로 추정 비용을 누적하고 이번 호출분을 반환."""
        pin = int(usage.get("promptTokenCount", 0) or 0)
        pout = int(usage.get("candidatesTokenCount", 0) or 0)
        pthink = int(usage.get("thoughtsTokenCount", 0) or 0)
        cost = (
            pin / 1e6 * _PRICE_IN_PER_M
            + (pout + pthink) / 1e6 * _PRICE_OUT_PER_M
            + searches * _PRICE_PER_SEARCH
        )
        with self._lock:
            self.searches_today += searches
            self.cost_today_usd += cost
        return cost

    def search_and_answer(self, question: str, context: dict | None = None) -> dict:
        question = (question or "").strip()
        if not question:
            return web_result(answer="", provider=self.name, enabled=True,
                              note="빈 질문 — 웹검색을 건너뜁니다.")
        if not self._take_budget():
            logger.warning("[web] 일일 웹검색 한도(%d회) 초과 — 호출하지 않습니다.", self._daily_budget)
            return web_result(answer="", provider=self.name, enabled=True,
                              note=f"일일 웹검색 한도({self._daily_budget}회)를 초과해 호출하지 않았습니다.")

        system = (context or {}).get("system") or self._system
        payload = {
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "tools": [{"google_search": {}}],
            "systemInstruction": {"parts": [{"text": system}]},
            # 사고(thinking) 토큰도 maxOutputTokens 를 함께 소진한다 → 답변이 잘려 빈 문자열이
            # 되지 않도록 여유 있게 잡는다(RAG 백엔드와 동일한 2048).
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
        }
        t0 = time.time()
        try:
            data = self._post(payload)
        except Exception as e:  # noqa: BLE001 - 위로 던지지 않는다
            logger.warning("[web] grounding 호출 실패(%s) — 웹검색 없이 보류합니다.", e)
            return web_result(answer="", provider=self.name, enabled=True,
                              note=f"웹검색 호출 실패({e})")

        res = self._parse(data)
        res["elapsed_s"] = round(time.time() - t0, 3)
        return res

    # ---------- 파싱 ----------
    def _parse(self, data: dict) -> dict:
        cand = (data.get("candidates") or [{}])[0] or {}
        parts = (cand.get("content") or {}).get("parts") or []
        answer = "".join(p.get("text", "") for p in parts).strip()

        gm = cand.get("groundingMetadata") or {}
        queries = [q for q in (gm.get("webSearchQueries") or []) if q]
        sources: list[dict] = []
        seen: set[str] = set()
        for chunk in gm.get("groundingChunks") or []:
            web = (chunk or {}).get("web") or {}
            url = web.get("uri")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({"title": web.get("title") or url, "url": url})
            if len(sources) >= self._max_sources:
                break

        usage = data.get("usageMetadata") or {}
        cost = self._record(len(queries), usage)

        if answer and not sources:
            # 검색을 전혀 타지 않은(모델 내부지식) 답변은 근거가 없다 → 채택하지 않는다.
            logger.warning("[web] 근거 출처 없음 — 웹 답변을 채택하지 않습니다.")
            return web_result(answer="", provider=self.name, enabled=True,
                              note="웹 검색 근거를 확보하지 못해 답변을 보류했습니다.")

        note = (f"Gemini Grounding({self.model}) · 검색 {len(queries)}회 · 출처 {len(sources)}건"
                if answer else "웹 검색에서도 답변을 찾지 못했습니다.")
        res = web_result(answer=answer, provider=self.name, enabled=True,
                         sources=sources, note=note)
        res.update(
            model=self.model,
            search_queries=queries,
            # Google 검색 grounding 이용약관은 응답과 함께 '검색 추천'(searchEntryPoint) 표시를
            # 요구한다 → 그대로 실어 보내고 프론트가 렌더한다.
            search_entry_point=(gm.get("searchEntryPoint") or {}).get("renderedContent"),
            usage={
                "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
                "output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
                "thought_tokens": int(usage.get("thoughtsTokenCount", 0) or 0),
                "searches": len(queries),
                "est_cost_usd": round(cost, 6),
            },
        )
        return res

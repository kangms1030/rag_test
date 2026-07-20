"""Gemini 2.5 Flash 백엔드 (P2) — 생성·검증만 Flash, 임베딩·리랭커는 로컬 불변.

원본 인터페이스: rag3.models.Backend (chat_text/chat_vision_text/embed). controller/answer/verify는
이 인터페이스만 알므로 컨트롤러 로직 무수정으로 끼워진다.

설계(프롬프트 §4 P2):
- embed(): 로컬 OllamaBackend에 위임 → 검색 품질 0.774(로컬 스택) 절대 불변, API 영구종속 회피.
- chat_text/chat_vision_text(): Gemini REST(generativelanguage.googleapis.com). 신규 의존성 0
  (requests만 사용). temp 0(결정론). Gemma 전용 방어(retry_on_length/num_predict)는 미적용.
- 안정성: API 타임아웃 · 429/5xx 지수 백오프 · 네트워크/재시도 소진 시 **로컬 12b 폴백**
  (전 구간 로컬로 돌아가는 길을 항상 유지).
- 비용: 호출별 입력/출력 토큰(usageMetadata) 기록 → 질문당 비용 실측(metrics + 선택적 JSONL).
- 키: .env의 GEMINI_API_KEY만 사용. **키를 로그/보고서/콘솔/예외 메시지에 절대 노출 안 함.**
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

from rag3.config import Config
from rag3.models import Backend, OllamaBackend, _resize_for_vlm

logger = logging.getLogger(__name__)

_API_HOST = "https://generativelanguage.googleapis.com/v1beta/models"

# 요율(USD / 1M tokens) — **가정치**. gemini-3.1-flash-lite(lite tier) 공개요율로 반드시 검증(무료면 0).
# 토큰 수(usageMetadata)는 정확하므로, 최종 비용표에는 "실측 토큰 × 가정 요율"로 표기하고
# 요율이 미확정임을 명시한다. (lite tier 가정치)
_PRICE_IN_PER_M = 0.10
_PRICE_OUT_PER_M = 0.40


def _load_api_key() -> str:
    """.env의 GEMINI_API_KEY 로드. 값은 절대 로깅/반환-노출하지 않는다."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        # 프로젝트 루트 .env 직접 파싱(python-dotenv 없이도 동작)
        try:
            from dotenv import dotenv_values
            root = Path(__file__).resolve().parents[2]  # .../챗봇
            key = (dotenv_values(root / ".env").get("GEMINI_API_KEY") or "").strip()
        except Exception:
            pass
    if not key:
        raise RuntimeError("GEMINI_API_KEY를 .env에서 찾지 못함 (값은 로깅하지 않음)")
    return key


class GeminiBackend(Backend):
    """생성/검증=Gemini Flash, 임베딩=로컬 위임. 실패 시 로컬 12b 폴백."""

    #: 프로세스 전역 최소 호출 간격 준수용(무료 20 RPM). 클래스 변수로 인스턴스 간 공유.
    _last_call_ts: float = 0.0

    def __init__(self, config: Config):
        import requests  # 신규 의존성 없음(기설치 확인)
        self._requests = requests
        self.config = config
        self._key = _load_api_key()
        self._local = OllamaBackend(config)  # 임베딩 위임 + 폴백용
        # 임베딩이 로컬 embeddinggemma로 동일하므로 flat 인덱스를 공유해야 한다
        # (flat_index는 backend_id로 인덱스 디렉터리를 키잉함). 로컬과 같은 id를 써야
        # 기존 인덱스(flat_chunk/ollama-embeddinggemma)를 찾는다.
        self.backend_id = self._local.backend_id
        self._model = getattr(config, "x_gemini_model", "gemini-3.1-flash-lite")
        self._timeout = int(getattr(config, "x_gemini_timeout_s", 60))
        self._max_retries = int(getattr(config, "x_gemini_max_retries", 4))
        self._fallback_local = bool(getattr(config, "x_gemini_fallback_local", True))
        self._vision_on = bool(getattr(config, "x_gemini_vision", False))
        self._cost_log = getattr(config, "x_cost_log_path", None)
        self._min_interval = float(getattr(config, "x_gemini_min_interval_s", 3.2))

    # --- 임베딩: 로컬 불변 위임 ---
    def embed(self, texts, *, is_query: bool = False):
        return self._local.embed(texts, is_query=is_query)

    # --- 비용/토큰 계측 ---
    def _record_usage(self, usage: dict, *, kind: str) -> None:
        pin = int(usage.get("promptTokenCount", 0) or 0)
        pout = int(usage.get("candidatesTokenCount", 0) or 0)
        cost = pin / 1e6 * _PRICE_IN_PER_M + pout / 1e6 * _PRICE_OUT_PER_M
        from rag3 import metrics
        m = metrics.current()
        if m is not None:
            acc = getattr(m, "_gemini", None)
            if acc is None:
                acc = {"in": 0, "out": 0, "cost": 0.0, "calls": 0}
                setattr(m, "_gemini", acc)  # controller_x가 결과 dict에 surface
            acc["in"] += pin
            acc["out"] += pout
            acc["cost"] += cost
            acc["calls"] += 1
        if self._cost_log:
            try:
                with open(self._cost_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"kind": kind, "in": pin, "out": pout,
                                        "cost": round(cost, 6)}, ensure_ascii=False) + "\n")
            except Exception:
                pass

    def _generation_config(self) -> dict:
        cfg = {"temperature": 0.0, "maxOutputTokens": 2048}
        tb = getattr(self.config, "x_gemini_thinking_budget", None)
        if tb is not None:
            cfg["thinkingConfig"] = {"thinkingBudget": int(tb)}
        return cfg

    def _throttle(self) -> None:
        """무료 20 RPM 준수: 직전 호출로부터 min_interval 경과 보장(전역). 429 예방."""
        if self._min_interval <= 0:
            return
        wait = self._min_interval - (time.time() - GeminiBackend._last_call_ts)
        if wait > 0:
            time.sleep(wait)
        GeminiBackend._last_call_ts = time.time()

    def _record_api_latency(self, sec: float) -> None:
        """raw API 왕복 지연 누적(스로틀 sleep 제외 = 순수 모델 속도)."""
        from rag3 import metrics
        m = metrics.current()
        if m is not None:
            acc = getattr(m, "_gemini", None)
            if acc is None:
                acc = {"in": 0, "out": 0, "cost": 0.0, "calls": 0, "api_s": 0.0}
                setattr(m, "_gemini", acc)
            acc["api_s"] = acc.get("api_s", 0.0) + sec

    def _post(self, payload: dict) -> dict:
        """generateContent 1회 호출 + 429/5xx 지수 백오프. 키는 URL 파라미터, 로그 비노출."""
        url = f"{_API_HOST}/{self._model}:generateContent"
        last_exc = None
        for attempt in range(self._max_retries + 1):
            try:
                self._throttle()
                _t = time.time()
                resp = self._requests.post(
                    url, params={"key": self._key},
                    json=payload, timeout=self._timeout,
                    headers={"Content-Type": "application/json"},
                )
                self._record_api_latency(time.time() - _t)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt, 30)
                    logger.warning("[gemini] %s → %ss 백오프(재시도 %d/%d)",
                                   resp.status_code, wait, attempt + 1, self._max_retries)
                    time.sleep(wait)
                    last_exc = RuntimeError(f"HTTP {resp.status_code}")
                    continue
                # 4xx(키/요청 오류): 본문에 키가 없도록 상태코드만 표면화
                raise RuntimeError(f"Gemini HTTP {resp.status_code}")
            except Exception as e:  # 네트워크/타임아웃
                last_exc = e
                wait = min(2 ** attempt, 30)
                logger.warning("[gemini] 네트워크 오류(%s) → %ss 백오프(%d/%d)",
                               type(e).__name__, wait, attempt + 1, self._max_retries)
                time.sleep(wait)
        raise RuntimeError(f"Gemini 호출 실패(재시도 소진): {type(last_exc).__name__}")

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            cand = data["candidates"][0]
            parts = cand.get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        except Exception:
            return ""

    def _chat(self, parts: list[dict], system: str | None, *, kind: str,
              local_fallback_call) -> str:
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "generationConfig": self._generation_config()}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            data = self._post(payload)
        except Exception as e:
            if self._fallback_local:
                logger.warning("[gemini] 폴백→로컬 12b (%s)", type(e).__name__)
                return local_fallback_call()
            raise
        self._record_usage(data.get("usageMetadata", {}), kind=kind)
        text = self._extract_text(data)
        if not text and self._fallback_local:
            logger.warning("[gemini] 빈 응답 → 로컬 12b 폴백")
            return local_fallback_call()
        return text

    def chat_text(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        return self._chat([{"text": prompt}], system, kind="text",
                          local_fallback_call=lambda: self._local.chat_text(prompt, model=model, system=system))

    def chat_vision_text(self, prompt: str, image_paths: list[Path], *,
                         model: str | None = None, system: str | None = None) -> str:
        # x_gemini_vision=False면 vision은 로컬(가설4 실험 격리). True면 Flash 멀티모달.
        if not self._vision_on:
            return self._local.chat_vision_text(prompt, image_paths, model=model, system=system)
        parts: list[dict] = [{"text": prompt}]
        for p in image_paths:
            b = _resize_for_vlm(p, self.config.answer_image_max_side)
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": base64.b64encode(b).decode()}})
        return self._chat(parts, system, kind="vision",
                          local_fallback_call=lambda: self._local.chat_vision_text(prompt, image_paths, model=model, system=system))

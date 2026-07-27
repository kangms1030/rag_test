"""실험 설정 로더 — 기존 rag3.config.load_config를 재사용하고 실험 플래그만 부착한다.

기존 rag3 코드는 신규 파라미터를 전부 `getattr(config, 'flag', default)`로 읽으므로
(models.py·retrieve.py·controller.py 참고), Config dataclass를 **수정하지 않고** 인스턴스에
setattr로 실험 플래그를 붙이면 된다. 전 플래그의 기본값은 "무동작(원본과 동일)"이다.

원본: rag3/config.py (Config, load_config) — 무수정 재사용.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rag3.config import Config, load_config

# 실험 플래그 기본값 — 전부 OFF/중립이라 rag3x가 rag3와 동일 동작(등가성 계약).
X_DEFAULTS: dict[str, Any] = {
    # --- 백엔드 선택 (P2) ---
    "x_backend": "ollama",           # "ollama"(기존) | "gemini"(생성·검증만 Flash, 임베딩은 로컬)
    # 계획은 gemini-2.5-flash 지정했으나 2026-07-19 현재 이 API 키(신규 사용자)에는 404
    # ("no longer available to new users") — 모든 variant·endpoint 확인. 2.0-flash 계열은
    # 429("check your plan")로 무료 불가. 사용자 결정(비용·한도 우선): 최저가·최고한도 lite tier
    # gemini-3.1-flash-lite(stable) 채택. 풀 Flash 대비 사고력↓ 가능 → 품질은 실측 보고.
    "x_gemini_model": "gemini-3.1-flash-lite",
    "x_gemini_timeout_s": 60,
    "x_gemini_max_retries": 4,       # 429/5xx 지수 백오프 재시도 횟수
    # 무료 티어 20 RPM(실측) 준수용 프로세스 전역 최소 호출 간격(초). 20rpm→3.2s.
    # 이 sleep은 raw API 지연과 분리 계측한다(속도 판정은 raw 기준 + 스로틀 오버헤드 별도 보고).
    "x_gemini_min_interval_s": 3.2,
    "x_gemini_fallback_local": True, # 네트워크/재시도 소진 시 로컬 12b 폴백
    "x_gemini_vision": False,        # 가설4: vision 경로도 Flash로
    # 2.5 Flash는 기본 thinking ON(지연·비용↑). 답변/검증 경로는 속도·결정론 위해 0(비활성).
    # 종합추론(P3)에서 사고력 필요 시 상향해 ablation. None이면 API 기본값(dynamic thinking).
    "x_gemini_thinking_budget": 0,

    # --- P1 로컬 속도 ---
    "x_fail_fast_on_length_budget": False,  # (a) length-retry 예산 소진 문항은 후속 롤백 생략, 즉시 확정
    "x_conditional_verify_skip": False,     # (b) 숫자대조통과∧고점∧단일문서면 groundedness LLM 생략
    "x_verify_skip_tau": 0.6,               # (b) 조건부 스킵 발동 rerank top 하한
    "x_adaptive_trim": False,               # (c) rerank 점수 급락 페이지를 컨텍스트에서 제외
    "x_adaptive_trim_drop_ratio": 0.5,      # (c) top 점수 대비 이 비율 미만 페이지 컷

    # --- P3 다문서 종합·추론 ---
    "x_enable_decompose_routing": False,    # 복합 질문일 때만 분해검색 경로 진입
    "x_decompose_max_subq": 4,
    "x_sentence_citation_verify": False,    # 합성답변 문장단위 인용검증(미지원 문장만 제거)

    # --- 계측 ---
    "x_cost_log_path": None,                # 호출별 토큰/비용 JSONL 경로(None이면 미기록)
}


def load_x_config(path: str | Path | None = None,
                  overrides: dict[str, Any] | None = None,
                  x_overrides: dict[str, Any] | None = None) -> Config:
    """rag3 config 로드 후 실험 플래그 부착. x_overrides로 실험 플래그만 개별 조정."""
    config = load_config(path, overrides)
    for k, v in X_DEFAULTS.items():
        setattr(config, k, v)
    if x_overrides:
        for k, v in x_overrides.items():
            if v is not None:
                setattr(config, k, v)
    return config

"""chatbot_demo_v2 설정.

pydantic-settings 미설치 환경이므로 frozen dataclass + os.environ + python-dotenv 로
설정을 구성한다. 환경변수 로딩 우선순위(먼저 로드된 쪽이 우선; override=False):

    프로세스 env > chatbot_demo_v2/.env > 최상위 .env > 코드 기본값

최상위 .env는 GEMINI_API_KEY 통과용으로만 읽고 수정하지 않는다.
API 키 값은 이 모듈에서 절대 로깅/출력하지 않는다.

v1(chatbot_demo) 대비 추가: RAGCORE(vendored) 경로, clarify/composer/contextualize/grader
토글, RAG 캐시 TTL, 프롬프트 폴더.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# chatbot_demo_v2 패키지 루트 (이 파일: chatbot_demo_v2/config/settings.py)
PKG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PKG_ROOT.parent  # 최상위 '챗봇' 폴더

_TRUE = {"1", "true", "yes", "on", "y", "t"}


def _load_dotenv_files() -> None:
    """chatbot_demo_v2/.env, 최상위 .env 순서로 로드(기존 값 보존)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        return
    load_dotenv(PKG_ROOT / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get(env: Mapping[str, str], name: str, default: str) -> str:
    val = env.get(name)
    if val is None or val == "":
        return default
    return val


def _get_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    val = env.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in _TRUE


def _get_float(env: Mapping[str, str], name: str, default: float) -> float:
    val = env.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_int(env: Mapping[str, str], name: str, default: int) -> int:
    val = env.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # rag(rag3x) 연동 — vendored ragcore 사용
    ragcore_root: Path            # chatbot_demo_v2/ragcore (sys.path 삽입 대상)
    ragcore_config: Path          # ragcore/rag3/config.yaml
    rag_backend: str              # "gemini" | "ollama"
    rag_deep_warmup: bool

    # 유사도 매칭
    scenario_match_threshold: float
    scenario_match_margin: float

    # clarify(HITL)
    clarify_enabled: bool
    clarify_min_score: float

    # composer / contextualize / grader
    composer_rag_enabled: bool
    composer_faq_enabled: bool
    contextualize_enabled: bool
    grader_enabled: bool

    # RAG 결과 캐시
    rag_cache_ttl_s: int

    # 웹검색
    web_search_enabled: bool
    web_search_scope: str         # "in_domain_unresolved" | "any_unresolved"

    # LangSmith
    langsmith_tracing: bool
    langsmith_project: str
    langsmith_endpoint: str
    langsmith_api_key_present: bool

    # 서버
    demo_port: int

    # 경로 (패키지 위치 기준 파생)
    data_dir: Path
    static_dir: Path
    prompts_dir: Path
    evidence_root: Path
    ragdata_dir: Path             # 복사 데이터 루트(index/parsed_v25)

    # GEMINI 키 존재 여부(값 미저장)
    gemini_api_key_present: bool = field(default=False)

    @property
    def faq_path(self) -> Path:
        return self.data_dir / "faq.json"

    @property
    def scenarios_path(self) -> Path:
        return self.data_dir / "scenarios.json"

    @property
    def faq_doc_links_path(self) -> Path:
        return self.data_dir / "faq_doc_links.json"

    @property
    def parsed_dir(self) -> Path:
        """파싱 캐시(페이지 이미지/표 크롭) 루트. faq 근거 이미지 해석에 사용."""
        return self.ragdata_dir / "parsed_v25"


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """설정을 로드한다.

    env를 주입하면(테스트) dotenv 로딩을 건너뛰고 해당 매핑만 사용한다.
    """
    if env is None:
        _load_dotenv_files()
        env = os.environ

    ragcore_root = Path(_get(env, "RAGCORE_ROOT", str(PKG_ROOT / "ragcore")))
    ragcore_config = Path(
        _get(env, "RAGCORE_CONFIG", str(ragcore_root / "rag3" / "config.yaml"))
    )

    data_dir = Path(_get(env, "DEMO_DATA_DIR", str(PKG_ROOT / "data")))
    static_dir = Path(_get(env, "DEMO_STATIC_DIR", str(PKG_ROOT / "static")))
    prompts_dir = Path(_get(env, "PROMPTS_DIR", str(PKG_ROOT / "prompts")))
    evidence_root = Path(
        _get(env, "DEMO_EVIDENCE_DIR", str(PKG_ROOT / "runtime" / "evidence"))
    )
    ragdata_dir = Path(_get(env, "RAGDATA_DIR", str(PKG_ROOT / "ragdata")))

    return Settings(
        ragcore_root=ragcore_root,
        ragcore_config=ragcore_config,
        rag_backend=_get(env, "RAG_BACKEND", "gemini"),
        rag_deep_warmup=_get_bool(env, "RAG_DEEP_WARMUP", False),
        scenario_match_threshold=_get_float(env, "SCENARIO_MATCH_THRESHOLD", 0.90),
        scenario_match_margin=_get_float(env, "SCENARIO_MATCH_MARGIN", 0.05),
        clarify_enabled=_get_bool(env, "CLARIFY_ENABLED", True),
        clarify_min_score=_get_float(env, "CLARIFY_MIN_SCORE", 0.75),
        composer_rag_enabled=_get_bool(env, "COMPOSER_RAG_ENABLED", True),
        composer_faq_enabled=_get_bool(env, "COMPOSER_FAQ_ENABLED", True),
        contextualize_enabled=_get_bool(env, "CONTEXTUALIZE_ENABLED", True),
        grader_enabled=_get_bool(env, "GRADER_ENABLED", True),
        rag_cache_ttl_s=_get_int(env, "RAG_CACHE_TTL_S", 3600),
        web_search_enabled=_get_bool(env, "WEB_SEARCH_ENABLED", False),
        web_search_scope=_get(env, "WEB_SEARCH_SCOPE", "in_domain_unresolved"),
        langsmith_tracing=_get_bool(env, "LANGSMITH_TRACING", False),
        langsmith_project=_get(env, "LANGSMITH_PROJECT", "school-network-chatbot-demo-v2"),
        langsmith_endpoint=_get(env, "LANGSMITH_ENDPOINT", ""),
        langsmith_api_key_present=bool(env.get("LANGSMITH_API_KEY")),
        demo_port=_get_int(env, "DEMO_PORT", 8002),
        data_dir=data_dir,
        static_dir=static_dir,
        prompts_dir=prompts_dir,
        evidence_root=evidence_root,
        ragdata_dir=ragdata_dir,
        gemini_api_key_present=bool(env.get("GEMINI_API_KEY")),
    )

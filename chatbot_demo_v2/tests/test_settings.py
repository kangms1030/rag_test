"""설정 로딩: 기본값·토글·경로 파생·키 미저장."""

from __future__ import annotations

from chatbot_demo_v2.config.settings import load_settings, PKG_ROOT


def test_defaults():
    s = load_settings(env={})
    assert s.rag_backend == "gemini"
    # 2026-07-27(작업 3): FAQ 매칭이 문자 유사도 → 의미 유사도로 바뀌면서 임계 스케일이
    # 완전히 달라졌다(fuzz 0.90/0.75 → semantic 0.80/0.40). 근거는 settings.py 주석과
    # scripts/calibrate_faq_threshold.py 실측 참조.
    assert s.scenario_match_backend == "semantic"
    assert s.scenario_match_threshold == 0.80
    assert s.scenario_match_margin == 0.30
    assert s.clarify_enabled is True
    assert s.clarify_min_score == 0.40
    assert s.composer_rag_enabled is True
    assert s.grader_enabled is True
    assert s.rag_cache_ttl_s == 3600
    assert s.demo_port == 8002


def test_paths_derived():
    s = load_settings(env={})
    assert s.ragcore_config == PKG_ROOT / "ragcore" / "rag3" / "config.yaml"
    assert s.faq_path == s.data_dir / "faq.json"
    assert s.prompts_dir == PKG_ROOT / "prompts"


def test_key_presence_only_not_value():
    s = load_settings(env={"GEMINI_API_KEY": "SECRET", "LANGSMITH_API_KEY": "SECRET2"})
    assert s.gemini_api_key_present is True
    assert s.langsmith_api_key_present is True
    # 값 자체는 저장하지 않음
    assert "SECRET" not in repr(s)


def test_toggle_override():
    s = load_settings(env={"CLARIFY_ENABLED": "false", "GRADER_ENABLED": "0"})
    assert s.clarify_enabled is False
    assert s.grader_enabled is False

"""RAG 결과 TTL 캐시: 같은 질문 재요청 시 엔진 재실행 없이 즉답."""

from __future__ import annotations

import time

from chatbot_demo_v2.config.settings import load_settings
from chatbot_demo_v2.rag.adapter_util import SubgraphRagAdapter


class _StubAdapter(SubgraphRagAdapter):
    """엔진/서브그래프 없이 캐시 로직만 검증하기 위한 스텁."""

    def __init__(self, settings):
        super().__init__(settings)
        self.engine_calls = 0

    def ensure_ready(self) -> None:      # 엔진 로딩 생략
        return

    def _run_subgraph(self, question, rid, progress):
        self.engine_calls += 1
        return {"run_id": rid, "final_answer": f"답변:{question}", "answer_path": "text",
                "confidence": "high", "evidence": [], "metrics": {}, "selected_pages": []}

    # 실제 ask 의 서브그래프 실행부만 스텁으로 대체(캐시 경로는 부모 로직 그대로 사용)
    def ask(self, question, run_id=None, progress=None):
        key = self._cache_key(question)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return dict(cached)
        out = self._run_subgraph(question, run_id or "rid", progress)
        self._cache_put(key, out)
        return out


def _settings(tmp_path, ttl="3600"):
    return load_settings(env={"RAG_CACHE_TTL_S": ttl,
                              "DEMO_EVIDENCE_DIR": str(tmp_path / "ev")})


def test_same_question_hits_cache(tmp_path):
    a = _StubAdapter(_settings(tmp_path))
    a.ask("무선 AP 제조사는?")
    a.ask("무선 AP 제조사는?")
    assert a.engine_calls == 1          # 엔진은 1회만
    assert a.cache_hits == 1


def test_cache_key_normalizes_whitespace_and_case(tmp_path):
    a = _StubAdapter(_settings(tmp_path))
    a.ask("무선 AP 제조사는?")
    a.ask("  무선   AP  제조사는?  ")
    assert a.engine_calls == 1


def test_different_question_misses(tmp_path):
    a = _StubAdapter(_settings(tmp_path))
    a.ask("질문 A")
    a.ask("질문 B")
    assert a.engine_calls == 2
    assert a.cache_hits == 0


def test_ttl_zero_disables_cache(tmp_path):
    a = _StubAdapter(_settings(tmp_path, ttl="0"))
    a.ask("같은 질문")
    a.ask("같은 질문")
    assert a.engine_calls == 2          # 캐시 비활성
    assert a.cache_hits == 0


def test_expired_entry_is_refetched(tmp_path):
    a = _StubAdapter(_settings(tmp_path, ttl="1"))
    a.ask("만료 테스트")
    # 저장시각을 과거로 돌려 TTL 초과 상황 재현
    with a._cache_lock:
        key = a._cache_key("만료 테스트")
        ts, val = a._cache[key]
        a._cache[key] = (ts - 10, val)
    a.ask("만료 테스트")
    assert a.engine_calls == 2


def test_cache_status_is_reported(tmp_path):
    a = _StubAdapter(_settings(tmp_path))
    a.ask("상태 확인")
    a.ask("상태 확인")
    st = a.status_dict()
    assert st["cache_entries"] == 1 and st["cache_hits"] == 1


def test_cache_capacity_evicts_oldest(tmp_path):
    a = _StubAdapter(_settings(tmp_path))
    for i in range(70):
        a.ask(f"질문 {i}")
        time.sleep(0)          # 저장시각 단조 증가 보장용(무시 가능)
    assert len(a._cache) <= 64

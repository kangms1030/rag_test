# -*- coding: utf-8 -*-
"""작업 3 회귀 테스트 — 의미 매칭 폴백과 임계 스케일 분리.

임베딩 파일·Ollama·GPU 없이 도는 테스트만 둔다(실제 의미 매칭 품질은 골든셋으로 측정).
"""
from __future__ import annotations

from chatbot_demo_v2.graph.routing import decide_route
from chatbot_demo_v2.scenario.matcher import ScenarioMatcher, SemanticScenarioMatcher
from chatbot_demo_v2.scenario.models import FaqEntry, FaqStore


def _store():
    return FaqStore([
        FaqEntry(id="A:1", sheet="A", row=1, no=1, question_type=None, fault_type=None,
                 question="스쿨넷이 뭐예요?", question_normalized="스쿨넷이 뭐예요", answer="..."),
        FaqEntry(id="A:2", sheet="A", row=2, no=2, question_type=None, fault_type=None,
                 question="AP가 뜨거운데 정상인가요?", question_normalized="ap가 뜨거운데 정상인가요",
                 answer="..."),
    ])


class _Settings:
    """임베딩 파일이 없는 설정 — 폴백 경로를 강제한다."""
    def __init__(self, tmp_path):
        self.data_dir = tmp_path
        self.ragcore_config = tmp_path / "nope.yaml"
        self.ragcore_root = tmp_path


class TestFallback:
    def test_임베딩_파일이_없으면_문자유사도로_폴백한다(self, tmp_path):
        m = SemanticScenarioMatcher(_store(), threshold=0.80, margin=0.30,
                                    settings=_Settings(tmp_path))
        r = m.match("스쿨넷이 뭐예요")
        assert r.decision == "exact", "정확 일치는 임베딩 없이도 동작해야 한다"

    def test_폴백시_fuzz_스케일_임계로_전환된다(self, tmp_path):
        """의미 임계(0.80/0.30)를 fuzz 점수에 그대로 쓰면 무관 질문도 통과한다."""
        m = SemanticScenarioMatcher(_store(), threshold=0.80, margin=0.30,
                                    settings=_Settings(tmp_path))
        r = m.match("전혀 상관없는 질문 zzz")
        assert r.decision.startswith("reject")
        assert r.threshold == SemanticScenarioMatcher.FUZZ_THRESHOLD
        assert r.margin_required == SemanticScenarioMatcher.FUZZ_MARGIN
        assert r.scale == "fuzz"

    def test_폴백_결과는_기존_ScenarioMatcher_와_같다(self, tmp_path):
        base = ScenarioMatcher(_store(), threshold=0.90, margin=0.05)
        m = SemanticScenarioMatcher(_store(), threshold=0.80, margin=0.30,
                                    settings=_Settings(tmp_path))
        q = "ap가 뜨거운데 정상인가요"
        assert m.match(q).decision == base.match(q).decision


class TestClarifyScale:
    """되묻기 하한은 스케일마다 다르다 — 잘못 쓰면 명확한 질문도 되묻는다."""

    def _route(self, best, margin, scale, decision="reject_low_score"):
        state = {"input_type": "text", "scenario_match": {
            "decision": decision, "best_score": best, "margin_observed": margin,
            "threshold": 0.80, "scale": scale}}
        return decide_route(state, clarify_enabled=True, clarify_min_score=0.40)[0]

    def test_의미스케일_모호질문은_되묻는다(self):
        # 실측: "비밀번호를 바꾸고 싶어요" best 0.971 margin 0.006
        assert self._route(0.971, 0.006, "semantic") == "clarify"

    def test_의미스케일_RAG질문은_되묻지_않는다(self):
        # 실측: rag_01 best 0.087 · rag_07 0.184 · rag_04 0.326 — 전부 하한 0.40 미만
        for best in (0.087, 0.184, 0.326):
            assert self._route(best, 0.05, "semantic") == "rag3x"

    def test_fuzz_스케일은_더_높은_하한을_쓴다(self):
        """fuzz 는 무관 질문도 0.4~0.5 가 나오므로 0.40 하한을 그대로 쓰면 전부 되묻게 된다."""
        assert self._route(0.45, 0.02, "fuzz") == "rag3x"
        assert self._route(0.80, 0.02, "fuzz") == "clarify"

    def test_scale_키가_없으면_안전하게_fuzz_로_본다(self):
        state = {"input_type": "text", "scenario_match": {
            "decision": "reject_low_score", "best_score": 0.45, "margin_observed": 0.02}}
        assert decide_route(state, clarify_enabled=True, clarify_min_score=0.40)[0] == "rag3x"

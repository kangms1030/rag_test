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
            "decision": decision, "best_score": best, "second_score": best - margin,
            "margin_observed": margin, "threshold": 0.80, "scale": scale}}
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


class TestClarifyNeedsTwoCandidates:
    """2026-08-04 — 후보가 사실상 1건이면 되묻지 않는다.

    되묻기 화면은 "비슷한 문의가 여러 건 있어요"라고 말한다. 2위가 0.00 인데 그 문구를
    띄우면 사용자에게 거짓말이 된다(실사용 제보). 점수는 전부 골든셋 실측값이다.
    """

    def _route(self, best, second, decision="reject_low_score", floor=0.80):
        state = {"input_type": "text", "scenario_match": {
            "decision": decision, "best_score": best, "second_score": second,
            "margin_observed": best - second, "threshold": 0.80, "scale": "semantic"}}
        return decide_route(state, clarify_enabled=True, clarify_min_score=floor)[0]

    def test_2위가_바닥이면_되묻지_않는다(self):
        # 제보 사례: "무선 AP 배치 시 2.4GHz/5GHz 채널 간섭…" — 1위만 애매하게 높다.
        assert self._route(0.642, 0.002) == "rag3x"
        assert self._route(0.563, 0.133) == "rag3x"   # rag_04 자가진단 체크리스트
        assert self._route(0.351, 0.000) == "rag3x"   # para_04

    def test_회색지대_점수는_되묻지_않는다(self):
        """0.65~0.78 은 골든셋 표본이 0개인 빈 구간 — 여기 걸리면 키워드만 겹친 것이다."""
        assert self._route(0.928, 0.644) == "rag3x"   # para_08 "어느 통신사?" ↔ "누가 설치?"
        assert self._route(0.780, 0.613) == "rag3x"   # ambig_04 "속도가 너무 느린데요"

    def test_후보_둘_다_정답급이면_되묻는다(self):
        assert self._route(0.942, 0.889, "reject_ambiguous") == "clarify"   # ambig_02
        assert self._route(0.971, 0.964, "reject_ambiguous") == "clarify"   # ambig_03
        assert self._route(0.931, 0.891, "reject_ambiguous") == "clarify"   # para_10

    def test_자동채택은_영향받지_않는다(self):
        assert self._route(0.99, 0.01, "accept") == "faq"

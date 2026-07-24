"""유사도 매처: 정확일치/높은유사도 채택/낮은유사도·애매 거부."""

from __future__ import annotations

from chatbot_demo.scenario.matcher import ScenarioMatcher, normalize_text
from chatbot_demo.scenario.models import FaqEntry, FaqStore


def _store():
    entries = [
        FaqEntry("s:2", "스쿨넷", 2, 1, "일반질문", "개념",
                 "스쿨넷이 뭐예요?", normalize_text("스쿨넷이 뭐예요?"),
                 "스쿨넷은 학교 인터넷 회선입니다.", ["a.pdf"]),
        FaqEntry("s:3", "스쿨넷", 3, 2, "장애", "연결",
                 "인터넷이 15분 넘게 안 돼요", normalize_text("인터넷이 15분 넘게 안 돼요"),
                 "회선 장애일 수 있습니다.", ["b.pdf"]),
        FaqEntry("s:4", "무선망", 4, 3, "장애", "연결",
                 "와이파이 목록에 학교 SSID가 안 보여요",
                 normalize_text("와이파이 목록에 학교 SSID가 안 보여요"),
                 "AP 신호가 꺼졌을 수 있습니다.", ["c.pdf"]),
    ]
    return FaqStore(entries)


def _matcher(threshold=0.90, margin=0.05):
    return ScenarioMatcher(_store(), threshold=threshold, margin=margin)


def test_exact_match():
    m = _matcher()
    r = m.match(normalize_text("스쿨넷이 뭐예요?"))
    assert r.decision == "exact"
    assert r.accepted and r.best_score == 1.0
    assert r.matched_sheet == "스쿨넷" and r.matched_row == 2


def test_normalization_whitespace_punct_case():
    m = _matcher()
    # 앞뒤 공백/구두점/대소문자/반복 공백 정리 후 정확 일치
    r = m.match(normalize_text("  스쿨넷이   뭐예요??  "))
    assert r.decision == "exact"


def test_low_score_rejected_goes_to_rag():
    m = _matcher()
    r = m.match(normalize_text("오늘 점심 메뉴 추천해줘"))
    assert not r.accepted
    assert r.decision == "reject_low_score"


def test_high_similarity_accepts():
    # 거의 동일하지만 완전 일치는 아닌 질문(어미만 다름) → 임계값 통과 시 accept.
    # 한국어 char 기반 fuzz.ratio 특성상 0.70 수준에서 채택되도록 임계값 설정.
    m = _matcher(threshold=0.70, margin=0.05)
    r = m.match(normalize_text("스쿨넷이 뭔가요"))
    assert r.best_score >= 0.70
    assert r.margin_observed >= 0.05
    assert r.decision == "accept"
    assert r.matched_row == 2


def test_ambiguous_margin_rejected():
    # 서로 매우 비슷한 두 후보를 만들어 margin 이 작아 애매 → 거부
    entries = [
        FaqEntry("x:2", "A", 2, 1, None, None, "인터넷이 안 돼요",
                 normalize_text("인터넷이 안 돼요"), "답변1", []),
        FaqEntry("x:3", "A", 3, 2, None, None, "인터넷이 안 되요",
                 normalize_text("인터넷이 안 되요"), "답변2", []),
    ]
    m = ScenarioMatcher(FaqStore(entries), threshold=0.5, margin=0.30)
    r = m.match(normalize_text("인터넷이 안 됩니다"))
    # 두 후보 점수가 비슷 → margin 부족으로 애매 거부
    assert r.decision == "reject_ambiguous"
    assert not r.accepted

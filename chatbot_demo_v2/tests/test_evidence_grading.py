# -*- coding: utf-8 -*-
"""작업 1·2·6 회귀 테스트 — abstain 오탐 / 근거 3등급 / 컨텍스트 예산 / CRAG 가드.

전부 LLM·GPU 없이 도는 순수 로직 테스트다(백엔드·프롬프트는 페이크 주입).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT / "ragcore"))

from rag3.verify import is_abstain  # noqa: E402


# ---------------------------------------------------------------- 작업 1
class TestIsAbstain:
    """LangSmith 8e0815cd: 정상 답변이 '제공된 근거' 때문에 회피로 오판됐다."""

    def test_정상답변_제공된근거에_따르면_으로_시작해도_회피가_아니다(self):
        # 트레이스에서 실제로 오판된 답변(발췌)
        ans = (
            "제공된 근거에 따르면, 학교에서 새로운 AP 장비를 추가하는 방법은 다음과 같습니다.\n\n"
            "통합관제 시스템의 '엑셀업로드' 기능을 통해 여러 대의 AP 장비를 한 번에 추가할 수 "
            "있습니다. 1. 템플릿 내려받기 2. 엑셀 작성 3. 엑셀 업로드 4. 목록 확인 및 제어"
        )
        assert is_abstain(ans) is False

    def test_프롬프트가_지시한_회피문구는_여전히_잡는다(self):
        assert is_abstain("제공된 근거에서 확인할 수 없습니다") is True

    def test_빈답변은_회피(self):
        assert is_abstain("") is True
        assert is_abstain("   \n ") is True

    def test_짧은_확인불가류는_회피(self):
        assert is_abstain("선택된 문서에서 확인 불가") is True
        assert is_abstain("해당 내용은 찾을 수 없습니다.") is True

    def test_긴답변_끝에_부분유보가_붙어도_회피가_아니다(self):
        ans = (
            "AP 장비는 통합관제 시스템의 엑셀업로드 기능으로 여러 대를 한 번에 등록할 수 있습니다. "
            "템플릿을 내려받아 교육청·지원청·학교명·AP명·IP·MAC 등을 작성한 뒤 업로드하면 됩니다. "
            "업로드 후에는 MAC중복체크 버튼으로 중복 여부를 확인하세요. "
            "다만 구체적인 승인 절차는 제공된 근거에서 확인할 수 없습니다."
        )
        assert len(ans) > 160
        assert is_abstain(ans) is False


# ---------------------------------------------------------------- 공용 픽스처
class _FakeBackend:
    """chat_text 만 쓰는 페이크. 반환값을 미리 지정한다."""

    def __init__(self, reply: str = ""):
        self.reply = reply
        self.calls: list[str] = []

    def chat_text(self, prompt, **kw):
        self.calls.append(prompt)
        return self.reply

    def embed(self, texts, **kw):
        return [[0.0] * 8 for _ in texts]


class _FakePrompts:
    def render(self, name, **kw):
        return f"[{name}] " + " ".join(f"{k}={v}" for k, v in kw.items())

    def load(self, name):
        class _T:
            template = ""
        return _T()

    def meta(self, name):
        return {}


def _page(doc, num, text, score, chunks=None, ptype="text"):
    return {
        "document_name": doc, "page_number": num, "page_score": score,
        "page_type": ptype, "is_scanned": False, "has_table": False,
        "figure_area_ratio": 0.0, "table_markdown": "", "table_crop_path": "",
        "page_image_path": "", "text": text,
        "matched_chunks": chunks or [],
    }


@pytest.fixture
def nodes_and_routers():
    """실제 make_rag_nodes 를 페이크 의존성으로 만든다."""
    from chatbot_demo_v2.graph.rag_nodes import RagDeps, make_rag_nodes

    class _Cfg:
        deadline_seconds = 240
        enable_crag = True
        enable_rollback = True
        enable_verify = False          # verify 는 이 테스트 범위 밖
        crag_retry_floor = 0.02
        rerank_score_floor = 0.1
        rollback_rerank_tau_high = 0.5
        context_max_chars = 10000
        figure_area_ratio_threshold = 0.5

    deps = RagDeps(config=_Cfg(), backend=_FakeBackend(), prompts=_FakePrompts(), scratch={})
    return make_rag_nodes(deps), deps


# ---------------------------------------------------------------- 작업 2: 등급 파싱
class TestGradeParsing:
    def test_등급_파싱과_irrelevant_제외(self, nodes_and_routers):
        (nodes, routers), deps = nodes_and_routers
        deps.backend.reply = "1=irrelevant\n2=primary\n3=supporting"
        pages = [_page("A.pdf", 22, "장애 조치", 0.57),
                 _page("B.pdf", 53, "엑셀업로드로 AP 추가", 0.35),
                 _page("A.pdf", 16, "SWIMS 조회", 0.29)]
        out = nodes["grade_evidence"]({"question": "q", "retrieval": {"selected_pages": pages}})

        assert [d["grade"] for d in out["grade_detail"]] == ["irrelevant", "primary", "supporting"]
        # irrelevant 는 제외되고 나머지는 _grade 를 달고 남는다
        assert [p["page_number"] for p in out["graded_pages"]] == [53, 16]
        assert out["graded_pages"][0]["_grade"] == "primary"

    def test_파싱실패시_전부_supporting_으로_안전하게_유지(self, nodes_and_routers):
        (nodes, routers), deps = nodes_and_routers
        deps.backend.reply = "무슨 말인지 모르겠습니다"
        pages = [_page("A.pdf", 1, "x", 0.5), _page("B.pdf", 2, "y", 0.4)]
        out = nodes["grade_evidence"]({"question": "q", "retrieval": {"selected_pages": pages}})
        assert [d["grade"] for d in out["grade_detail"]] == ["supporting", "supporting"]
        assert len(out["graded_pages"]) == 2, "판정 실패가 근거 유실로 이어지면 안 된다"

    def test_근거가_없으면_빈결과(self, nodes_and_routers):
        (nodes, _), _ = nodes_and_routers
        out = nodes["grade_evidence"]({"question": "q", "retrieval": {"selected_pages": []}})
        assert out["graded_pages"] == []


# ---------------------------------------------------------------- 작업 2: 라우팅
class TestAfterGrade:
    def test_남은근거_있으면_answer(self, nodes_and_routers):
        (_, routers), _ = nodes_and_routers
        assert routers["after_grade"]({"graded_pages": [_page("A", 1, "x", 0.5)]}) == "answer"

    def test_전부_irrelevant_이고_예산_있으면_crag(self, nodes_and_routers):
        (_, routers), _ = nodes_and_routers
        assert routers["after_grade"]({"graded_pages": [], "crag_budget": 1}) == "crag"

    def test_전부_irrelevant_이고_예산_없으면_no_answer(self, nodes_and_routers):
        (_, routers), _ = nodes_and_routers
        assert routers["after_grade"]({"graded_pages": [], "crag_budget": 0}) == "no_answer"

    def test_after_retrieve_는_grade_로_보낸다(self, nodes_and_routers):
        (_, routers), _ = nodes_and_routers
        state = {"retrieval": {"answer_path": "text", "rerank_top_score": 0.5}}
        assert routers["after_retrieve"](state) == "grade"


# ---------------------------------------------------------------- 작업 2: 예산 배분
class TestBudget:
    """실측 문제: 컨텍스트 5,949자 중 정답 9.6%, 무관 3순위가 64.6%, 한 문서가 90.1% 독점."""

    def _budget(self, nodes_and_routers, graded):
        (nodes, _), _ = nodes_and_routers
        state = {"retrieval": {"selected_pages": [], "answer_path": "text"},
                 "graded_pages": graded, "question": "q"}
        # answer_node 를 태우지 않고 예산 함수만 검증하기 위해 answer_node 의 결과를 본다
        out = nodes["answer_node"](state)
        return out["budgeted_pages"]

    def test_primary_는_전문_supporting_은_축약(self, nodes_and_routers):
        long_primary = "P" * 3000
        long_support = "S" * 3000
        graded = [dict(_page("정답.pdf", 53, long_primary, 0.35), _grade="primary"),
                  dict(_page("노이즈.pdf", 16, long_support, 0.29), _grade="supporting")]
        pages = self._budget(nodes_and_routers, graded)
        by = {p["page_number"]: len(p["text"]) for p in pages}
        assert by[53] == 3000, "primary 는 전문이 들어가야 한다"
        assert by[16] <= 800, "supporting 은 상한(800)을 넘으면 안 된다"
        assert by[53] > by[16], "정답 근거가 노이즈보다 많이 들어가야 한다"

    def test_supporting_총합_상한(self, nodes_and_routers):
        graded = [dict(_page("정답.pdf", 1, "P" * 500, 0.5), _grade="primary")]
        for i in range(2, 8):
            graded.append(dict(_page(f"보조{i}.pdf", i, "S" * 3000, 0.3), _grade="supporting"))
        pages = self._budget(nodes_and_routers, graded)
        sup = sum(len(p["text"]) for p in pages if p.get("_grade") == "supporting")
        assert sup <= 2000, f"supporting 합계 상한 초과: {sup}"

    def test_한_문서가_supporting_예산을_독점하지_못한다(self, nodes_and_routers):
        graded = [dict(_page("정답.pdf", 1, "P" * 300, 0.5), _grade="primary")]
        for pg in (10, 11, 12, 13):
            graded.append(dict(_page("독점.pdf", pg, "S" * 3000, 0.3), _grade="supporting"))
        pages = self._budget(nodes_and_routers, graded)
        hog = sum(len(p["text"]) for p in pages if p["document_name"] == "독점.pdf")
        assert hog <= 2000 * 0.6 + 1, f"한 문서가 supporting 예산을 독점했다: {hog}"

    def test_primary_가_없으면_supporting_최상위를_승격한다(self, nodes_and_routers):
        """그레이딩이 primary 를 못 찾아도 답변을 통째로 버리지 않는 안전장치."""
        # supporting 상한(800)과 primary 상한(4000)을 구분할 수 있도록 2,000자로 둔다
        graded = [dict(_page("A.pdf", 1, "본" * 2000, 0.4), _grade="supporting"),
                  dict(_page("B.pdf", 2, "본" * 2000, 0.3), _grade="supporting")]
        pages = self._budget(nodes_and_routers, graded)
        assert pages, "primary 가 없다고 근거를 전부 버리면 안 된다"
        assert pages[0].get("_promoted") is True
        assert len(pages[0]["text"]) == 2000, "승격된 페이지는 primary 예산(전문)을 받아야 한다"
        # 나머지는 supporting 대우
        assert all(len(p["text"]) <= 800 for p in pages[1:])

    def test_발단_사례_재현_정답비중이_올라간다(self, nodes_and_routers):
        """LangSmith 8e0815cd 구성을 그대로 넣어 정답 근거 비중을 확인한다."""
        graded = [
            dict(_page("무선랜가이드.pdf", 22, "장" * 1396, 0.5701), _grade="supporting"),
            dict(_page("통합관제.pdf", 53, "엑" * 574, 0.3508), _grade="primary"),
            # p16(3845자, 64.6% 차지하던 무관 페이지)은 irrelevant 로 이미 제외됐다고 가정
        ]
        pages = self._budget(nodes_and_routers, graded)
        total = sum(len(p["text"]) for p in pages)
        answer_chars = sum(len(p["text"]) for p in pages if p["page_number"] == 53)
        share = answer_chars / total
        assert share > 0.40, f"정답 근거 비중이 여전히 낮다: {share:.1%} (기준선 9.6%)"


# ---------------------------------------------------------------- 작업 6: CRAG 가드
class TestCragGuard:
    """실측: 질의 재작성이 정답 근거를 1위 → 7위로 악화시켰다."""

    def _retrieve_with(self, nodes_and_routers, monkeypatch, new_top, new_path="text"):
        (nodes, _), deps = nodes_and_routers
        import chatbot_demo_v2.graph.rag_nodes as rn  # noqa: F401  (모듈 존재 확인용)
        from rag3.retrieve import RetrievalResult

        def fake_run_retrieval(q, config, backend):
            return RetrievalResult(q, [{"document_name": "N.pdf"}],
                                   [_page("N.pdf", 9, "새 결과", new_top)],
                                   answer_path=new_path, route_reason="fake",
                                   rerank_top_score=new_top)

        monkeypatch.setattr("rag3.retrieve.run_retrieval", fake_run_retrieval)
        # make_rag_nodes 가 import 시점에 바인딩하므로 새로 만든다
        from chatbot_demo_v2.graph.rag_nodes import make_rag_nodes
        nodes2, _ = make_rag_nodes(deps)
        prev = {"selected_pages": [_page("O.pdf", 1, "원래 결과", 0.55)],
                "selected_documents": [{"document_name": "O.pdf"}],
                "answer_path": "text", "route_reason": "orig", "rerank_top_score": 0.55,
                "candidate_chunks": []}
        return nodes2["retrieve"]({"question": "q", "rewritten_query": "재작성", "retrieval": prev})

    def test_재작성이_더_나쁘면_원본을_유지한다(self, nodes_and_routers, monkeypatch):
        out = self._retrieve_with(nodes_and_routers, monkeypatch, new_top=0.21)
        assert out["retrieval"]["rerank_top_score"] == 0.55, "더 나쁜 재작성 결과를 채택했다"
        assert out["history"][0]["action"] == "crag_rewrite_rejected"

    def test_재작성이_더_좋으면_채택한다(self, nodes_and_routers, monkeypatch):
        out = self._retrieve_with(nodes_and_routers, monkeypatch, new_top=0.80)
        assert out["retrieval"]["rerank_top_score"] == 0.80
        assert out["history"][0]["action"] == "crag_rewrite_accepted"

    def test_재작성이_근거없음이면_원본을_유지한다(self, nodes_and_routers, monkeypatch):
        out = self._retrieve_with(nodes_and_routers, monkeypatch, new_top=0.9, new_path="none")
        assert out["retrieval"]["rerank_top_score"] == 0.55
        assert out["history"][0]["action"] == "crag_rewrite_rejected"


# ---------------------------------------------------------------- 작업 5: 계측
def test_RagState_에_started_ts_가_선언돼_있다():
    """미선언 시 LangGraph 가 조용히 버려 timings.total 이 항상 0.0 이 됐다."""
    from chatbot_demo_v2.graph.state import RagState
    assert "started_ts" in RagState.__annotations__
    assert "graded_pages" in RagState.__annotations__
    assert "budgeted_pages" in RagState.__annotations__

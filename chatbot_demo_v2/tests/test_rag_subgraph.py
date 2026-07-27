"""RAG 서브그래프: CRAG 사이클·롤백 트리거·no_answer 경로 (원시함수 fake 주입).

원시함수(run_retrieval/answer/verify/…)를 fake 로 대체해 GPU·LLM 없이 그래프 로직만 검증한다.
등가성(서브그래프 vs 원본 엔진)은 RUN_RAG_INTEGRATION=1 일 때만 도는 통합 테스트로 분리.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PKG = Path(__file__).resolve().parents[1]
RAGCORE = PKG / "ragcore"
if str(RAGCORE) not in sys.path:
    sys.path.insert(0, str(RAGCORE))

# ragcore 가 없거나(=미부트스트랩) import 실패 시 이 모듈 전체 skip
rag3 = pytest.importorskip("rag3")
import rag3.retrieve as _retrieve_mod
import rag3.answer as _answer_mod
import rag3.controller as _controller_mod
import rag3.judge as _judge_mod
import rag3x.answer_x as _answerx_mod
import rag3x.controller_x as _ctrlx_mod
from rag3.retrieve import RetrievalResult
from rag3 import metrics as _metrics

from chatbot_demo_v2.graph.builder import build_rag_subgraph
from chatbot_demo_v2.graph.rag_nodes import RagDeps


def _fake_config(**over):
    base = dict(
        deadline_seconds=240,
        enable_crag=True,
        enable_rollback=True,
        enable_verify=True,
        crag_retry_floor=0.02,
        rerank_score_floor=0.1,
        rollback_rerank_tau_high=0.5,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _page(text="근거 텍스트", scanned=False):
    return {
        "document_name": "doc.pdf", "page_number": 1, "text": text,
        "page_image_path": "", "table_crop_path": "", "is_scanned": scanned,
        "page_type": "text", "figure_area_ratio": 0.0,
    }


def _rr(path="text", top=0.8, pages=None):
    pages = pages if pages is not None else [_page()]
    return RetrievalResult(
        question="q", selected_documents=[{"document_name": "doc.pdf"}],
        selected_pages=pages, answer_path=path, route_reason="fake", rerank_top_score=top,
    )


@pytest.fixture
def patch_primitives(monkeypatch):
    """원시함수를 fake 로 교체. 반환된 dict 로 각 fake 동작 제어."""
    ctrl = {
        "retrievals": [_rr()],          # run_retrieval 이 순서대로 반환
        "answer": {"final_answer": "정상 답변입니다.", "context": "근거 텍스트", "raw": ""},
        "verify": {"abstain": False, "unsupported_claims": [], "transcription_ocr_mismatch": [],
                   "confidence": "high", "deterministic_ok": True, "groundedness": "supported"},
        "rewritten": "재작성된 질문",
        "calls": {"retrieve": 0, "answer_text": 0, "answer_vision": 0, "verify": 0, "rewrite": 0},
    }

    def fake_retrieve(q, config, backend):
        i = ctrl["calls"]["retrieve"]
        ctrl["calls"]["retrieve"] += 1
        seq = ctrl["retrievals"]
        return seq[min(i, len(seq) - 1)]

    def fake_answer_text(q, pages, backend, config):
        ctrl["calls"]["answer_text"] += 1
        a = ctrl["answer"]
        return dict(a) if not callable(a) else a(pages)

    def fake_answer_vision(q, page, backend, config):
        ctrl["calls"]["answer_vision"] += 1
        return {"final_answer": "비전 답변", "context": "", "raw": "", "transcription": ""}

    def fake_verify(ans, path, page0, backend, config, *, ground):
        ctrl["calls"]["verify"] += 1
        v = ctrl["verify"]
        return dict(v) if not callable(v) else v(ans, path)

    def fake_decide_ground(ans, path, top, docs, config):
        return True

    def fake_rewrite(q, backend, config):
        ctrl["calls"]["rewrite"] += 1
        return ctrl["rewritten"]

    monkeypatch.setattr(_retrieve_mod, "run_retrieval", fake_retrieve)
    monkeypatch.setattr(_answerx_mod, "answer_text_from_pages_x", fake_answer_text)
    monkeypatch.setattr(_answer_mod, "answer_vision_from_page", fake_answer_vision)
    monkeypatch.setattr(_controller_mod, "_verify", fake_verify)
    monkeypatch.setattr(_ctrlx_mod, "_decide_ground", fake_decide_ground)
    monkeypatch.setattr(_judge_mod, "rewrite_query", fake_rewrite)
    return ctrl


def _run(ctrl, config):
    deps = RagDeps(config=config, backend=object(), scratch={"metrics": _metrics.RunMetrics()})
    sub = build_rag_subgraph(deps)
    with _metrics.run_metrics(deps.scratch["metrics"]):
        return sub.invoke({"question": "질문", "run_id": "rid", "history": []})


def test_happy_path_text(patch_primitives):
    out = _run(patch_primitives, _fake_config())
    res = out["result"]
    assert res["final_answer"] == "정상 답변입니다."
    assert res["answer_path"] == "text"
    assert patch_primitives["calls"]["retrieve"] == 1
    assert patch_primitives["calls"]["rewrite"] == 0


def test_crag_cycle_once(patch_primitives):
    # 1차 검색 none(경계 점수) → crag_rewrite → 2차 검색 정상
    patch_primitives["retrievals"] = [_rr(path="none", top=0.05), _rr(path="text", top=0.8)]
    out = _run(patch_primitives, _fake_config())
    res = out["result"]
    assert patch_primitives["calls"]["rewrite"] == 1          # 재작성 1회
    assert patch_primitives["calls"]["retrieve"] == 2          # 재검색 1회(사이클 1회)
    assert res["answer_path"] == "text"
    assert any(h["action"] == "crag_rewrite" for h in res["rollback_history"])


def test_crag_budget_prevents_infinite_loop(patch_primitives):
    # 재검색도 계속 none → crag 예산 소진 후 no_answer (무한루프 없음)
    patch_primitives["retrievals"] = [_rr(path="none", top=0.05)]
    out = _run(patch_primitives, _fake_config())
    res = out["result"]
    assert patch_primitives["calls"]["rewrite"] == 1
    assert patch_primitives["calls"]["retrieve"] == 2          # 1 + crag 1회, 그 이상 없음
    assert res["answer_path"] == "none"
    assert res["final_answer"] == "선택된 문서에서 확인 불가"


def test_no_answer_no_crag_when_score_too_low(patch_primitives):
    # top 이 crag_retry_floor 미만(무관) → crag 안 함
    patch_primitives["retrievals"] = [_rr(path="none", top=0.0)]
    out = _run(patch_primitives, _fake_config())
    assert patch_primitives["calls"]["rewrite"] == 0
    assert patch_primitives["calls"]["retrieve"] == 1
    assert out["result"]["answer_path"] == "none"


def test_rollback_top1_on_abstain(patch_primitives):
    # text abstain + 고점 → rollback_top1 재시도, 2번째 답변은 정상
    answers = [
        {"final_answer": "확인 불가", "context": "c", "raw": ""},
        {"final_answer": "재시도 정상 답변", "context": "c", "raw": ""},
    ]
    seq = {"i": 0}

    def answer_seq(pages):
        a = answers[min(seq["i"], len(answers) - 1)]
        seq["i"] += 1
        return dict(a)

    patch_primitives["answer"] = answer_seq
    verifies = [
        {"abstain": True, "unsupported_claims": [], "transcription_ocr_mismatch": [], "confidence": "abstain"},
        {"abstain": False, "unsupported_claims": [], "transcription_ocr_mismatch": [], "confidence": "high"},
    ]
    vi = {"i": 0}

    def verify_seq(ans, path):
        v = verifies[min(vi["i"], len(verifies) - 1)]
        vi["i"] += 1
        return dict(v)

    patch_primitives["verify"] = verify_seq
    out = _run(patch_primitives, _fake_config())
    res = out["result"]
    assert patch_primitives["calls"]["answer_text"] == 2       # 최초 + 롤백 재시도
    assert res["final_answer"] == "재시도 정상 답변"
    assert any(h["action"] == "rollback_text_top1" for h in res["rollback_history"])


@pytest.mark.skipif(
    os.environ.get("RUN_RAG_INTEGRATION") != "1",
    reason="실기(GPU/Ollama/Gemini) 필요 — RUN_RAG_INTEGRATION=1 로 활성화",
)
def test_subgraph_equivalence_with_engine():
    """서브그래프 결과가 원본 Rag3xEngine.ask 와 핵심 필드에서 일치(비결정성 감안)."""
    from chatbot_demo_v2.config.settings import load_settings
    from chatbot_demo_v2.rag.adapter_util import SubgraphRagAdapter, prepare_ragcore_imports

    s = load_settings()
    prepare_ragcore_imports(s)
    from rag3x import Rag3xEngine

    engine = Rag3xEngine(config_path=str(s.ragcore_config),
                         x_overrides={"x_backend": s.rag_backend}, preload=True)
    adapter = SubgraphRagAdapter(s)

    for q in ["무선 AP는 어느 제조사들로 구축되어 있나요?",
              "스쿨넷 서비스는 무엇인가요?"]:
        base = engine.ask(q)
        sub = adapter.ask(q)
        assert (sub["answer_path"] == base["answer_path"]), f"path 불일치: {q}"
        # 근거 페이지 문서 집합 일치(순서/내용 비결정 최소화)
        base_docs = {e["document_name"] for e in (base.get("evidence") or [])}
        sub_docs = {e["document_name"] for e in (sub.get("evidence") or [])}
        assert base_docs == sub_docs, f"근거 문서 불일치: {q}"

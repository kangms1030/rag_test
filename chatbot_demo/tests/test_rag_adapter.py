"""rag3x 어댑터: 결과 정규화(절대경로 제거), busy/미가용 예외."""

from __future__ import annotations

import pytest

from chatbot_demo.rag.rag3x_adapter import (
    FakeRagAdapter,
    RagBusyError,
    RagUnavailableError,
    Rag3xAdapter,
    normalize_rag_result,
    _sanitize_error,
    STATUS_NOT_LOADED,
)


def test_normalize_strips_absolute_paths():
    raw = {
        "run_id": "run1",
        "final_answer": "답변",
        "answer_path": "vision",
        "confidence": "high",
        "rerank_top_score": 0.9,
        "evidence": [
            {
                "document_name": "doc.pdf",
                "page_number": 3,
                "page_image_path": r"C:\zzsecretzz\hidden\p3.png",
                "page_image_resolved": r"C:\zzsecretzz\hidden\p3.png",
                "table_crop_resolved": None,
            }
        ],
        "verification": {"confidence": "high", "abstain": False},
        "metrics": {"total_model_calls": 2, "timings_seconds": {"total": 1.0}},
    }
    captured = {}

    def fake_copy(item, rank):
        captured["run"] = item.get("_run_id")
        return {"image_url": "/evidence/run1/ev1_p0003.png", "table_url": None}

    out = normalize_rag_result(raw, "run1", copy_evidence=fake_copy)
    blob = str(out)
    assert "zzsecretzz" not in blob and "hidden" not in blob and "C:\\" not in blob
    assert out["evidence"][0]["image_url"] == "/evidence/run1/ev1_p0003.png"
    assert captured["run"] == "run1"


def test_sanitize_error_removes_paths_and_tokens():
    msg = _sanitize_error(RuntimeError(r"failed at C:\Users\x\key with token abcdefghij1234567890KEY"))
    assert "C:\\Users" not in msg
    assert "abcdefghij1234567890KEY" not in msg


def test_fake_adapter_busy_and_unavailable():
    busy = FakeRagAdapter(raise_busy=True)
    with pytest.raises(RagBusyError):
        busy.ask("q")

    unavail = FakeRagAdapter(raise_unavailable=True)
    with pytest.raises(RagUnavailableError):
        unavail.ask("q")


def test_real_adapter_lock_serializes(settings):
    """실제 어댑터의 ask 락: 동시 진입 시 두 번째는 RagBusyError.
    엔진 초기화까지 가지 않도록 lock 을 직접 점유해 검증."""
    adapter = Rag3xAdapter(settings)
    assert adapter.status == STATUS_NOT_LOADED
    # ask 내부 락을 외부에서 선점
    got = adapter._ask_lock.acquire(blocking=False)
    assert got
    try:
        with pytest.raises(RagBusyError):
            adapter.ask("동시 요청")
    finally:
        adapter._ask_lock.release()

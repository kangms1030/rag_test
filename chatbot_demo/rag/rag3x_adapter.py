"""rag3x 엔진 어댑터.

- rag3x/rag3 는 외부 모듈로 import(수정 금지). test_3/코드 를 sys.path 에 삽입.
- 엔진은 프로세스당 1개, 지연 초기화(상태기계). GEMINI_API_KEY 가 없으면 생성이 실패하므로
  시나리오/FAQ 기능이 키·GPU 없이도 동작하도록 rag 경로에서만 초기화한다.
- ask() 는 threading.Lock 으로 직렬화. 동시 요청은 RagBusyError.
- 근거 이미지는 rag3x 의 절대경로를 chatbot_demo/runtime/evidence/<run_id>/ 로 복사하고
  안전한 상대 URL(/evidence/<run_id>/<basename>)만 상태로 내보낸다. (test_3 에는 쓰지 않음)
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ..config.settings import Settings, PROJECT_ROOT

# 엔진 상태
STATUS_NOT_LOADED = "not_loaded"
STATUS_LOADING = "loading"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class RagBusyError(Exception):
    """이미 다른 RAG 질문을 처리 중일 때."""


class RagUnavailableError(Exception):
    """엔진 초기화 실패/미가용."""


def _sanitize_error(exc: BaseException) -> str:
    """예외 메시지에서 절대경로/비밀스러운 토큰을 제거한 안전한 요약."""
    msg = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    # 윈도우 절대경로 제거
    msg = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "<path>", msg)
    msg = re.sub(r"/[^\s'\"]+/[^\s'\"]+", "<path>", msg)
    # 키처럼 보이는 긴 토큰 제거
    msg = re.sub(r"[A-Za-z0-9_\-]{20,}", "<redacted>", msg)
    return f"{exc.__class__.__name__}: {msg}"[:300]


_RUNID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class Rag3xAdapter:
    """rag3x 엔진 지연 로딩 + 직렬화 + 근거 사본."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None
        self._status = STATUS_NOT_LOADED
        self._error: Optional[str] = None
        self._init_lock = threading.Lock()   # 초기화 경쟁 방지
        self._ask_lock = threading.Lock()    # 질문 직렬화(단일 GPU)
        self._deep_warmed = False

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def evidence_root(self) -> Path:
        return self._settings.evidence_root

    def status_dict(self) -> dict:
        return {"status": self._status, "error": self._error}

    # --- 초기화 ---
    def _prepare_imports(self) -> None:
        root = str(self._settings.rag3x_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        # GEMINI_API_KEY 를 최상위 .env 에서 os.environ 으로 통과(값 노출 금지).
        if not os.environ.get("GEMINI_API_KEY"):
            try:
                from dotenv import dotenv_values

                vals = dotenv_values(PROJECT_ROOT / ".env")
                key = vals.get("GEMINI_API_KEY")
                if key:
                    os.environ["GEMINI_API_KEY"] = key
            except Exception:
                pass

    def ensure_ready(self) -> None:
        """엔진을 생성(필요 시). 실패하면 RagUnavailableError."""
        if self._status == STATUS_READY:
            return
        with self._init_lock:
            if self._status == STATUS_READY:
                return
            self._status = STATUS_LOADING
            self._error = None
            try:
                self._prepare_imports()
                from rag3x import Rag3xEngine

                x_overrides = {"x_backend": self._settings.rag3x_backend}
                engine = Rag3xEngine(
                    config_path=str(self._settings.rag3x_config),
                    x_overrides=x_overrides,
                    preload=True,
                )
                self._engine = engine
                self._status = STATUS_READY
            except Exception as exc:  # noqa: BLE001
                self._status = STATUS_FAILED
                self._error = _sanitize_error(exc)
                raise RagUnavailableError(self._error) from None

    def warm_up(self, deep: bool = False) -> None:
        """엔진 준비 + (옵션) 딥 워밍업. 백그라운드 스레드에서 호출 가능."""
        self.ensure_ready()
        if deep and not self._deep_warmed and self._engine is not None:
            self._engine.warm_up(deep=True)
            self._deep_warmed = True

    def start_warmup_background(self, deep: bool = False) -> None:
        """비블로킹 워밍업 시작. 실패는 status/error 에 남긴다."""

        def _run():
            try:
                self.warm_up(deep=deep)
            except Exception:
                # 상태/에러는 ensure_ready 에서 이미 기록됨.
                pass

        t = threading.Thread(target=_run, name="rag3x-warmup", daemon=True)
        t.start()

    # --- 질의 ---
    def ask(self, question: str, run_id: Optional[str] = None) -> dict:
        """질문 1건 처리. 동시 요청이면 RagBusyError, 미가용이면 RagUnavailableError."""
        acquired = self._ask_lock.acquire(blocking=False)
        if not acquired:
            raise RagBusyError("이미 다른 질문을 처리 중입니다.")
        try:
            self.ensure_ready()
            raw = self._engine.ask(
                question,
                run_id=run_id,
                resolve_images=True,
                save_evidence=False,
            )
            rid = raw.get("run_id") or run_id or "run"
            return normalize_rag_result(raw, rid, copy_evidence=self._copy_evidence)
        finally:
            self._ask_lock.release()

    def _copy_evidence(self, item: dict, rank: int) -> dict:
        """근거 이미지 절대경로를 evidence_root/<run_id>/ 로 복사하고 URL 반환.

        반환: {image_url, table_url} (없으면 None).
        run_id 는 item 에 주입돼 있어야 함.
        """
        run_id = item.get("_run_id", "run")
        if not _RUNID_RE.match(run_id):
            run_id = "run"
        out: dict[str, Optional[str]] = {"image_url": None, "table_url": None}
        dest_dir = self._settings.evidence_root / run_id
        page = item.get("page_number")
        page_str = f"{int(page):04d}" if isinstance(page, int) else "0000"

        for key, suffix, out_key in (
            ("page_image_resolved", "", "image_url"),
            ("table_crop_resolved", "_table", "table_url"),
        ):
            src = item.get(key)
            if not src:
                continue
            src_path = Path(src)
            if not src_path.is_file():
                continue
            ext = src_path.suffix or ".png"
            fname = f"ev{rank}_p{page_str}{suffix}{ext}"
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_path, dest_dir / fname)
                out[out_key] = f"/evidence/{run_id}/{fname}"
            except Exception:
                out[out_key] = None
        return out


def normalize_rag_result(
    raw: dict,
    run_id: str,
    copy_evidence: Callable[[dict, int], dict],
) -> dict:
    """rag3x ask() 결과를 프론트/상태용 표준 dict 로 변환.

    - 절대 파일 경로는 제외하고 안전한 상대 URL 만 남긴다.
    - copy_evidence(item, rank) 는 {image_url, table_url} 를 반환.
    """
    metrics = raw.get("metrics") or {}
    timings = metrics.get("timings_seconds") or {}

    evidence_out: list[dict] = []
    for rank, ev in enumerate(raw.get("evidence") or [], start=1):
        item = dict(ev)
        item["_run_id"] = run_id
        urls = copy_evidence(item, rank)
        evidence_out.append(
            {
                "rank": rank,
                "document_name": ev.get("document_name"),
                "page_number": ev.get("page_number"),
                "image_url": urls.get("image_url"),
                "table_url": urls.get("table_url"),
            }
        )

    verification = raw.get("verification")
    verification_out = None
    if isinstance(verification, dict):
        verification_out = {
            "confidence": verification.get("confidence"),
            "abstain": bool(verification.get("abstain")),
            "unsupported_claims": verification.get("unsupported_claims"),
            "transcription_ocr_mismatch": verification.get("transcription_ocr_mismatch"),
        }

    return {
        "run_id": run_id,
        "final_answer": raw.get("final_answer"),
        "answer_path": raw.get("answer_path"),
        "confidence": raw.get("confidence"),
        "route_reason": raw.get("route_reason"),
        "rerank_top_score": raw.get("rerank_top_score"),
        "evidence": evidence_out,
        "verification": verification_out,
        "selected_pages": _strip_page_paths(raw.get("selected_pages") or []),
        "rollback_history": raw.get("rollback_history") or [],
        "metrics": {
            "total_model_calls": metrics.get("total_model_calls"),
            "text_answer_calls": metrics.get("text_answer_calls"),
            "vision_answer_calls": metrics.get("vision_answer_calls"),
            "rerank_calls": metrics.get("rerank_calls"),
            "judge_calls": metrics.get("judge_calls"),
            "verify_calls": metrics.get("verify_calls"),
            "length_retry_count": metrics.get("length_retry_count"),
            "timings_seconds": {
                "retrieve": timings.get("retrieve"),
                "answer": timings.get("answer"),
                "total": timings.get("total"),
            },
            "gemini_calls": metrics.get("gemini_calls"),
        },
    }


def _strip_page_paths(pages: list[dict]) -> list[dict]:
    """selected_pages 에서 절대경로 키를 제거한 요약."""
    out = []
    for p in pages:
        out.append(
            {
                "document_name": p.get("document_name"),
                "page_number": p.get("page_number"),
                "score": p.get("score") or p.get("rerank_score"),
            }
        )
    return out


class FakeRagAdapter:
    """테스트용 어댑터. 실제 엔진 없이 정해진 결과를 반환한다."""

    def __init__(
        self,
        result: dict | None = None,
        *,
        status: str = STATUS_READY,
        raise_busy: bool = False,
        raise_unavailable: bool = False,
    ):
        self._result = result
        self._status = status
        self._raise_busy = raise_busy
        self._raise_unavailable = raise_unavailable
        self.ask_calls = 0
        self.warmup_calls = 0
        self.evidence_root = Path(".")

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self):
        return None

    def status_dict(self) -> dict:
        return {"status": self._status, "error": None}

    def ensure_ready(self) -> None:
        if self._raise_unavailable:
            raise RagUnavailableError("테스트: 엔진 미가용")

    def warm_up(self, deep: bool = False) -> None:
        self.warmup_calls += 1
        self.ensure_ready()

    def start_warmup_background(self, deep: bool = False) -> None:
        self.warmup_calls += 1

    def ask(self, question: str, run_id: str | None = None) -> dict:
        self.ask_calls += 1
        if self._raise_busy:
            raise RagBusyError("테스트: 처리 중")
        if self._raise_unavailable:
            raise RagUnavailableError("테스트: 엔진 미가용")
        if self._result is not None:
            return dict(self._result)
        return {
            "run_id": run_id or "fake-run",
            "final_answer": "테스트 답변",
            "answer_path": "text",
            "confidence": "high",
            "route_reason": "fake",
            "rerank_top_score": 0.5,
            "evidence": [],
            "verification": {"confidence": "high", "abstain": False},
            "selected_pages": [],
            "rollback_history": [],
            "metrics": {"total_model_calls": 1, "timings_seconds": {"total": 0.01}},
        }

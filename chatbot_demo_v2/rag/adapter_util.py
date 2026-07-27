"""rag3x 엔진 어댑터 + 공용 유틸 (v2).

v1 chatbot_demo/rag/rag3x_adapter.py 이식. 변경점:
- vendored ragcore 를 sys.path 에 삽입(settings.ragcore_root) 후 `import rag3x`.
- settings 필드명 v2화(ragcore_root/ragcore_config/rag_backend).

이 파일이 제공하는 것:
- Rag3xAdapter: 엔진 지연초기화 + 직렬화(Lock) + 근거 이미지 사본 (Phase 1 blackbox 경로).
- normalize_rag_result / _copy_evidence / _sanitize_error: Phase 2 RAG 서브그래프 finalize 에서 재사용.
- FakeRagAdapter: 테스트용.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..config.settings import Settings, PROJECT_ROOT

# 엔진 상태
STATUS_NOT_LOADED = "not_loaded"
STATUS_LOADING = "loading"
STATUS_READY = "ready"
STATUS_FAILED = "failed"


class RagBusyError(Exception):
    """이미 다른 RAG 질문을 처리 중일 때."""


class RagUnavailableError(Exception):
    """엔진 초기화 실패/미가용."""


def _sanitize_error(exc: BaseException) -> str:
    """예외 메시지에서 절대경로/비밀스러운 토큰을 제거한 안전한 요약."""
    msg = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    msg = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "<path>", msg)
    msg = re.sub(r"/[^\s'\"]+/[^\s'\"]+", "<path>", msg)
    msg = re.sub(r"[A-Za-z0-9_\-]{20,}", "<redacted>", msg)
    return f"{exc.__class__.__name__}: {msg}"[:300]


_RUNID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def prepare_ragcore_imports(settings: Settings) -> None:
    """vendored ragcore 를 sys.path 에 넣고 GEMINI 키를 통과시킨다(값 노출 금지)."""
    root = str(settings.ragcore_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    if not os.environ.get("GEMINI_API_KEY"):
        try:
            from dotenv import dotenv_values

            vals = dotenv_values(PROJECT_ROOT / ".env")
            key = vals.get("GEMINI_API_KEY")
            if key:
                os.environ["GEMINI_API_KEY"] = key
        except Exception:
            pass


def make_evidence_copier(evidence_root: Path) -> Callable[[dict, int], dict]:
    """근거 이미지 사본 함수를 생성한다(evidence_root 바인딩).

    반환 함수 signature: copy_evidence(item: dict, rank: int) -> {image_url, table_url}.
    item 에는 '_run_id' 와 page_image_resolved/table_crop_resolved(절대경로)가 있어야 한다.
    """

    def _copy_evidence(item: dict, rank: int) -> dict:
        run_id = item.get("_run_id", "run")
        if not _RUNID_RE.match(run_id):
            run_id = "run"
        out: dict[str, Optional[str]] = {"image_url": None, "table_url": None}
        dest_dir = evidence_root / run_id
        page = item.get("page_number")
        try:
            page_str = f"{int(page):04d}"
        except (TypeError, ValueError):
            page_str = "0000"

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

    return _copy_evidence


def normalize_rag_result(
    raw: dict,
    run_id: str,
    copy_evidence: Callable[[dict, int], dict],
) -> dict:
    """rag3x ask() (또는 서브그래프 finalize) 결과를 프론트/상태용 표준 dict 로 변환.

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
        # composer(Phase 4) 근거용 페이지 텍스트(파일경로 없음). 없으면 빈 문자열.
        "answer_context": raw.get("answer_context") or "",
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


class Rag3xAdapter:
    """rag3x 엔진 지연 로딩 + 직렬화 + 근거 사본 (Phase 1 blackbox 경로)."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None
        self._status = STATUS_NOT_LOADED
        self._error: Optional[str] = None
        self._init_lock = threading.Lock()
        self._ask_lock = threading.Lock()
        self._deep_warmed = False
        self._copy_evidence = make_evidence_copier(settings.evidence_root)

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

    def ensure_ready(self) -> None:
        if self._status == STATUS_READY:
            return
        with self._init_lock:
            if self._status == STATUS_READY:
                return
            self._status = STATUS_LOADING
            self._error = None
            try:
                prepare_ragcore_imports(self._settings)
                from rag3x import Rag3xEngine

                engine = Rag3xEngine(
                    config_path=str(self._settings.ragcore_config),
                    x_overrides={"x_backend": self._settings.rag_backend},
                    preload=True,
                )
                self._engine = engine
                self._status = STATUS_READY
            except Exception as exc:  # noqa: BLE001
                self._status = STATUS_FAILED
                self._error = _sanitize_error(exc)
                raise RagUnavailableError(self._error) from None

    def warm_up(self, deep: bool = False) -> None:
        self.ensure_ready()
        if deep and not self._deep_warmed and self._engine is not None:
            self._engine.warm_up(deep=True)
            self._deep_warmed = True

    def start_warmup_background(self, deep: bool = False) -> None:
        def _run():
            try:
                self.warm_up(deep=deep)
            except Exception:
                pass

        t = threading.Thread(target=_run, name="rag3x-warmup", daemon=True)
        t.start()

    def ask(self, question: str, run_id: Optional[str] = None, progress=None) -> dict:
        # progress: 블랙박스 경로라 내부 단계를 알 수 없어 무시(시그니처 호환용).
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


class SubgraphRagAdapter:
    """프로덕션 RAG 어댑터 (Phase 2): rag3x.ask() 블랙박스 대신 RAG 서브그래프를 실행한다.

    .ask()/.status_dict()/.warm_up()/.start_warmup_background() 인터페이스는
    Rag3xAdapter/FakeRagAdapter 와 동일 → 메인그래프 rag3x_answer 노드·테스트 주입 무변경.

    엔진 구성(config+backend+warm_up)은 vendored Rag3xEngine 을 재사용하고, ask 는
    controller_x.answer_question(블랙박스) 대신 build_rag_subgraph 로 만든 서브그래프를
    with metrics.run_metrics(m) 안에서 invoke 한다 → LangSmith 에 각 노드가 child run 으로 보인다.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None
        self._deps = None
        self._subgraph = None
        self._status = STATUS_NOT_LOADED
        self._error: Optional[str] = None
        self._init_lock = threading.Lock()
        self._ask_lock = threading.Lock()
        self._deep_warmed = False
        self._copy_evidence = make_evidence_copier(settings.evidence_root)
        # 질문 → (저장시각, 정규화 결과) TTL 캐시. 같은 질문 재요청 시 25~150초를 아낀다.
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_lock = threading.Lock()
        self.cache_hits = 0

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
        return {"status": self._status, "error": self._error, "mode": "subgraph",
                "cache_entries": len(self._cache), "cache_hits": self.cache_hits}

    # --- TTL 캐시 ---
    def _cache_key(self, question: str) -> str:
        return " ".join((question or "").split()).lower()

    def _cache_get(self, key: str) -> Optional[dict]:
        ttl = int(self._settings.rag_cache_ttl_s or 0)
        if ttl <= 0:
            return None
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is None:
                return None
            ts, val = hit
            if time.time() - ts > ttl:
                self._cache.pop(key, None)
                return None
            return val

    def _cache_put(self, key: str, value: dict) -> None:
        ttl = int(self._settings.rag_cache_ttl_s or 0)
        if ttl <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.time(), value)
            if len(self._cache) > 64:      # 상한 — 가장 오래된 항목 제거
                oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
                self._cache.pop(oldest, None)

    def ensure_ready(self) -> None:
        if self._status == STATUS_READY:
            return
        with self._init_lock:
            if self._status == STATUS_READY:
                return
            self._status = STATUS_LOADING
            self._error = None
            try:
                prepare_ragcore_imports(self._settings)
                from rag3x import Rag3xEngine

                from ..graph.builder import build_rag_subgraph
                from ..graph.rag_nodes import RagDeps

                engine = Rag3xEngine(
                    config_path=str(self._settings.ragcore_config),
                    x_overrides={"x_backend": self._settings.rag_backend},
                    preload=True,
                )
                self._engine = engine
                self._deps = RagDeps(config=engine.config, backend=engine.backend, scratch={})
                self._subgraph = build_rag_subgraph(self._deps)
                self._status = STATUS_READY
            except Exception as exc:  # noqa: BLE001
                self._status = STATUS_FAILED
                self._error = _sanitize_error(exc)
                raise RagUnavailableError(self._error) from None

    def warm_up(self, deep: bool = False) -> None:
        self.ensure_ready()
        if deep and not self._deep_warmed and self._engine is not None:
            self._engine.warm_up(deep=True)
            self._deep_warmed = True

    def start_warmup_background(self, deep: bool = False) -> None:
        def _run():
            try:
                self.warm_up(deep=deep)
            except Exception:
                pass

        t = threading.Thread(target=_run, name="rag3x-subgraph-warmup", daemon=True)
        t.start()

    def ask(self, question: str, run_id: Optional[str] = None, progress=None) -> dict:
        """질문 1건 처리. progress(dict) 콜백을 주면 서브그래프 각 단계를 스트리밍한다."""
        # 캐시는 락 획득 전에 확인 — 다른 질문 처리 중이어도 캐시된 답변은 즉시 준다(429 회피).
        key = self._cache_key(question)
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            if progress is not None:
                try:
                    progress({"stage": "cache", "msg": "이전에 답변한 질문이라 저장된 결과를 씁니다."})
                except Exception:  # noqa: BLE001
                    pass
            return dict(cached)

        acquired = self._ask_lock.acquire(blocking=False)
        if not acquired:
            raise RagBusyError("이미 다른 질문을 처리 중입니다.")
        try:
            self.ensure_ready()
            from rag3 import metrics
            from rag3.utils import new_run_id

            rid = run_id or new_run_id(question)
            m = metrics.RunMetrics()
            self._deps.scratch["metrics"] = m
            self._deps.scratch["progress"] = progress
            init_state = {
                "question": question,
                "run_id": rid,
                "history": [],
            }
            with metrics.run_metrics(m):
                out = self._subgraph.invoke(init_state)
            result = out.get("result") or {}
            rid2 = result.get("run_id") or rid
            normalized = normalize_rag_result(result, rid2, copy_evidence=self._copy_evidence)
            self._cache_put(key, normalized)
            return normalized
        finally:
            self._ask_lock.release()


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

    def ask(self, question: str, run_id: str | None = None, progress=None) -> dict:
        self.ask_calls += 1
        if progress is not None:      # 스트리밍 테스트용 — 실제 단계는 없으므로 1건만
            try:
                progress({"stage": "retrieve", "msg": "(fake) 검색 중…"})
            except Exception:  # noqa: BLE001
                pass
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

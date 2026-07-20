"""모델을 호출하지 않는 환경 점검. `python -m rag2 check`.

의존성이 하나 없어도(예: chromadb) 나머지 항목은 계속 점검하고, 실패 이유를 사람이 바로
알 수 있게 출력한다. torch CUDA 가용성과 Chroma 컬렉션 count(REPORT B6: 손상된 빈 인덱스
조기감지)도 함께 확인한다.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

from .config import Config

_REQUIRED_MODULES = ["pandas", "openpyxl", "fitz", "chromadb", "ollama", "rank_bm25", "rapidfuzz", "yaml", "PIL", "torch"]
_OPTIONAL_MODULES = ["kiwipiepy", "mineru", "pdfplumber"]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _check_module(name: str) -> CheckResult:
    try:
        importlib.import_module(name)
        return CheckResult(name, True, "import OK")
    except ImportError as e:
        return CheckResult(name, False, f"MISSING ({e})")


def _check_torch_cuda() -> CheckResult:
    try:
        import torch

        if torch.cuda.is_available():
            return CheckResult("torch_cuda", True, f"{torch.__version__}, device={torch.cuda.get_device_name(0)}")
        return CheckResult("torch_cuda", False, f"torch {torch.__version__} 설치됐으나 CUDA 미가용 (CPU 폴백)")
    except ImportError as e:
        return CheckResult("torch_cuda", False, f"torch 미설치 ({e})")


def _check_ollama_server(config: Config) -> CheckResult:
    try:
        import ollama

        client = ollama.Client()
        resp = client.list()
    except ImportError:
        return CheckResult("ollama_server", False, "ollama 패키지 미설치")
    except Exception as e:
        return CheckResult("ollama_server", False, f"서버 연결 실패: {e}")

    installed = [m.model for m in resp.models]
    installed_base = {name.split(":")[0] for name in installed}

    def _has(required: str) -> bool:
        return required in installed or required.split(":")[0] in installed_base

    required_models = (config.text_answer_model, config.vision_answer_model, config.embedding_model, config.fallback_model)
    missing = [m for m in required_models if not _has(m)]
    if missing:
        return CheckResult("ollama_server", False, f"서버는 응답하나 모델 없음: {missing} (설치됨: {installed})")
    return CheckResult("ollama_server", True, f"연결 OK, 필요 모델 전부 설치됨: {installed}")


def _check_index_counts(config: Config) -> list[CheckResult]:
    try:
        from .index import get_index
        from .models import get_backend

        backend = get_backend(config)
        results = []
        for name in ("catalog_index", "page_index"):
            idx = get_index(name, config, backend)
            n = idx.count()
            results.append(CheckResult(f"{name}_count", n > 0, f"{n}개 레코드" if n > 0 else "비어 있음 (ingest 필요)"))
        return results
    except Exception as e:
        return [CheckResult("index_counts", False, f"인덱스 점검 실패: {e}")]


def check_environment(config: Config | None = None, *, check_indexes: bool = True) -> dict[str, Any]:
    results: list[CheckResult] = [CheckResult("python", True, sys.version.split()[0])]
    for m in _REQUIRED_MODULES:
        results.append(_check_module(m))
    for m in _OPTIONAL_MODULES:
        r = _check_module(m)
        r.name = f"{m} (optional)"
        results.append(r)
    results.append(_check_torch_cuda())

    if config is not None:
        results.append(_check_ollama_server(config))
        if check_indexes:
            results.extend(_check_index_counts(config))
    else:
        results.append(CheckResult("ollama_server", False, "config 로드 실패로 점검 생략"))

    required_results = [r for r in results if not r.name.endswith("(optional)") and r.name not in ("python", "ollama_server", "torch_cuda")]
    all_required_ok = all(r.ok for r in required_results)
    return {
        "all_required_ok": all_required_ok,
        "ollama_server_ok": next((r.ok for r in results if r.name == "ollama_server"), False),
        "torch_cuda_ok": next((r.ok for r in results if r.name == "torch_cuda"), False),
        "checks": [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in results],
    }

"""모델을 호출하지 않는 환경 점검. `python -m rag_catalog_experiment check-env`.

의존성이 하나 없어도(예: chromadb) 나머지 항목은 계속 점검하고, 실패 이유를 사람이 바로
알 수 있게 출력한다. 이 모듈 자체는 chromadb/pandas 등 무거운 의존성을 top-level에서
import하지 않는다 — 그래야 그것들이 없을 때도 check-env가 죽지 않는다.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

from .config import Config

_REQUIRED_MODULES = ["pandas", "openpyxl", "fitz", "pdfplumber", "chromadb", "ollama", "rank_bm25", "rapidfuzz", "yaml", "PIL"]
_OPTIONAL_MODULES = ["kiwipiepy", "pytesseract"]


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


def _check_ollama_server(config: Config) -> CheckResult:
    try:
        import ollama

        client = ollama.Client()
        resp = client.list()
    except ImportError:
        return CheckResult("ollama_server", False, "ollama 패키지 미설치")
    except Exception as e:
        return CheckResult("ollama_server", False, f"서버 연결 실패: {e}")

    # ollama.list()는 dict가 아니라 ListResponse(models=[Model(model="name:tag", ...), ...])를 반환한다.
    # config.yaml의 모델명은 태그 없이(예: "embeddinggemma") 적혀 있고 서버에는 ":latest"가 붙어
    # 있을 수 있으므로, 태그를 뗀 베이스 이름으로도 매칭해야 한다.
    installed = [m.model for m in resp.models]
    installed_base = {name.split(":")[0] for name in installed}

    def _has(required: str) -> bool:
        return required in installed or required.split(":")[0] in installed_base

    required_models = (config.llm_model, config.vlm_model, config.embedding_model, config.fallback_model)
    missing = [m for m in required_models if not _has(m)]
    if missing:
        return CheckResult("ollama_server", False, f"서버는 응답하나 모델 없음: {missing} (설치됨: {installed})")
    return CheckResult("ollama_server", True, f"연결 OK, 필요 모델 전부 설치됨: {installed}")


def check_environment(config: Config | None = None) -> dict[str, Any]:
    results: list[CheckResult] = [CheckResult("python", True, sys.version.split()[0])]
    for m in _REQUIRED_MODULES:
        results.append(_check_module(m))
    for m in _OPTIONAL_MODULES:
        r = _check_module(m)
        r.name = f"{m} (optional)"
        results.append(r)

    if config is not None:
        results.append(_check_ollama_server(config))
    else:
        results.append(CheckResult("ollama_server", False, "config 로드 실패로 점검 생략"))

    required_results = [r for r in results if not r.name.endswith("(optional)") and r.name not in ("python", "ollama_server")]
    all_required_ok = all(r.ok for r in required_results)
    return {
        "all_required_ok": all_required_ok,
        "ollama_server_ok": next((r.ok for r in results if r.name == "ollama_server"), False),
        "checks": [{"name": r.name, "ok": r.ok, "detail": r.detail} for r in results],
    }

"""rag3x — RAG3 Phase 6 병행 실험판 (기존 rag3 무수정).

원칙(프롬프트 §5·§8.1):
- 기존 rag3 패키지는 한 줄도 수정하지 않는다. 동작 불변부는 rag3에서 import 재사용.
- 변경부(controller/answer 변형, Gemini 백엔드, 분해검색)만 이 패키지에 명시적 포크로 둔다.
- 모든 신규 동작은 실험 플래그로 on/off. **전 플래그 OFF면 rag3와 완전 동일하게 동작**(등가성 계약).
- 공유 자원(인덱스·파싱캐시·카탈로그)은 읽기 전용. 산출물은 test_3/tmp/·probes/results/만.
"""
from __future__ import annotations

__all__ = ["Rag3xEngine", "load_x_config"]


def __getattr__(name: str):  # lazy: import rag3x 자체는 가볍게 유지
    if name == "Rag3xEngine":
        from .engine_x import Rag3xEngine
        return Rag3xEngine
    if name == "load_x_config":
        from .xconfig import load_x_config
        return load_x_config
    raise AttributeError(name)

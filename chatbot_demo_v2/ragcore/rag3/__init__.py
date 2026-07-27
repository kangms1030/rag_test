"""rag3 — 한국어 문서 RAG 파이프라인 (MinerU 파싱 + 하이브리드 검색 + 검증/롤백).

외부 연동 진입점:
    from rag3 import Rag3Engine, load_config
    engine = Rag3Engine()          # 프로세스당 1회 (모델 웜 로딩)
    result = engine.ask("질문")     # engine.AskResult 스키마 참고

CLI: python -m rag3 {check|ingest|add|ask|evaluate}  /  test_3/ask_cli.py (대화형)
"""
from .config import Config, load_config
from .engine import AskResult, EvidenceItem, Rag3Engine

__all__ = ["Config", "load_config", "Rag3Engine", "AskResult", "EvidenceItem"]

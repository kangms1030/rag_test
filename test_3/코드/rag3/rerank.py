"""bge-reranker-v2-m3 크로스인코더 래퍼 (Phase 0-D에서 채택).

질문-청크 쌍을 재점수화해 검색 후보를 재정렬한다. fp16 GPU 상주(~1.5GB, Phase 0-E 실측).
sentence-transformers CrossEncoder 사용. 프로세스 내 1회 로드 후 재사용.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Config

logger = logging.getLogger(__name__)

_RERANKER_CACHE: dict[str, "Reranker"] = {}


@dataclass
class RerankHit:
    index: int      # 입력 리스트에서의 원 인덱스
    score: float


class Reranker:
    def __init__(self, config: Config):
        from sentence_transformers import CrossEncoder
        import torch

        device = config.rerank_device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("rerank_device=cuda이나 CUDA 불가 -> cpu")
            device = "cpu"
        self.model = CrossEncoder(config.rerank_model, max_length=config.rerank_max_length, device=device)
        self.config = config

    def rank(self, query: str, docs: list[str]) -> list[RerankHit]:
        """docs를 점수 내림차순으로 정렬한 RerankHit 리스트 반환(원 인덱스 보존)."""
        if not docs:
            return []
        pairs = [(query, d[:6000]) for d in docs]  # 안전상 상한(리랭커 내부 토큰 truncation 별도)
        scores = self.model.predict(pairs)
        order = sorted(range(len(docs)), key=lambda i: -float(scores[i]))
        return [RerankHit(index=i, score=float(scores[i])) for i in order]


def get_reranker(config: Config) -> Reranker:
    key = config.rerank_model
    if key not in _RERANKER_CACHE:
        logger.info("리랭커 로드: %s", key)
        _RERANKER_CACHE[key] = Reranker(config)
    return _RERANKER_CACHE[key]

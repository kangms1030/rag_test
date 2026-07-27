"""청크 전용 경량 하이브리드 검색 (Chroma HNSW 비의존).

Chroma 1.5.9는 큰 컬렉션(수천건) HNSW 세그먼트를 별도 프로세스에서 reload할 때 재현적으로 실패한다
(B6: "Error loading hnsw index" — test_2/test_3 공통). 수천 청크 규모에선 근사최근접(HNSW)이 과하고
브루트포스 코사인이 정확·충분히 빠르다(2417x768 < 10ms). 따라서 청크 dense 검색은 numpy로 직접 하고,
BM25(kiwi)와 RRF로 융합한다. 벡터/문서/메타는 npz+json으로 디스크에 저장(오프라인 ingest -> 온라인 ask 재사용).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .index import ScoredItem
from .models import Backend
from .tokenizer import tokenize_ko

logger = logging.getLogger(__name__)


class FlatChunkIndex:
    def __init__(self, config: Config, backend: Backend):
        self.config = config
        self.backend = backend
        self.dir = config.index_dir / "flat_chunk" / backend.backend_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._loaded = False
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self._emb: np.ndarray | None = None      # (N, D) L2-정규화
        self._bm25 = None
        self._tokenized: list[list[str]] = []

    # --- 저장/로드 ---
    @property
    def _vec_path(self) -> Path:
        return self.dir / "vectors.npz"

    @property
    def _doc_path(self) -> Path:
        return self.dir / "docs.json"

    def build(self, ids: list[str], texts: list[str], metas: list[dict[str, Any]]) -> None:
        """임베딩 계산 후 디스크에 저장(기존 내용 대체)."""
        if not ids:
            return
        vecs = self.backend.embed(texts, is_query=False)
        arr = np.asarray(vecs, dtype=np.float32)
        arr = _l2norm(arr)
        np.savez(self._vec_path, ids=np.array(ids, dtype=object), emb=arr)
        self._doc_path.write_text(
            json.dumps({"ids": ids, "docs": texts, "metas": metas}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._loaded = False  # 다음 query에서 재로드
        logger.info("FlatChunkIndex: %d개 청크 저장 (%s)", len(ids), self.dir)

    def _load(self) -> None:
        if self._loaded:
            return
        if not self._vec_path.exists() or not self._doc_path.exists():
            self._ids, self._docs, self._metas, self._emb = [], [], [], None
            self._loaded = True
            return
        data = json.loads(self._doc_path.read_text(encoding="utf-8"))
        self._ids, self._docs, self._metas = data["ids"], data["docs"], data["metas"]
        npz = np.load(self._vec_path, allow_pickle=True)
        self._emb = npz["emb"].astype(np.float32)
        self._tokenized = [tokenize_ko(d, self.config.tokenizer) for d in self._docs]
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        self._loaded = True

    def count(self) -> int:
        self._load()
        return len(self._ids)

    # --- 증분 갱신 (문서 추가/교체용; build()와 동일한 npz+json 포맷 유지) ---
    def _save(self, ids: list[str], docs: list[str], metas: list[dict], emb: np.ndarray) -> None:
        if emb.shape[0] != len(ids):
            raise RuntimeError(
                f"인덱스 불일치: 벡터 {emb.shape[0]}개 vs id {len(ids)}개 — 저장 중단(풀 ingest 재실행 필요)")
        np.savez(self._vec_path, ids=np.array(ids, dtype=object), emb=emb.astype(np.float32))
        self._doc_path.write_text(
            json.dumps({"ids": ids, "docs": docs, "metas": metas}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._loaded = False  # BM25/토큰화는 다음 query에서 재구성

    def append(self, ids: list[str], texts: list[str], metas: list[dict[str, Any]]) -> int:
        """신규 청크만 임베딩해 기존 인덱스 뒤에 이어붙여 저장. 추가된 개수 반환.

        기존 벡터는 재임베딩하지 않는다. 같은 문서를 교체할 때는 remove_doc을 먼저 호출할 것
        (id 중복이면 인덱스 오염 방지를 위해 실패).
        """
        if not ids:
            return 0
        self._loaded = False
        self._load()
        dup = set(ids) & set(self._ids)
        if dup:
            raise ValueError(f"이미 색인된 chunk_id {len(dup)}개(예: {sorted(dup)[:3]}) — remove_doc 후 append 필요")
        vecs = self.backend.embed(texts, is_query=False)
        arr = _l2norm(np.asarray(vecs, dtype=np.float32))
        emb = arr if self._emb is None or not len(self._ids) else np.vstack([self._emb, arr])
        self._save(self._ids + list(ids), self._docs + list(texts), self._metas + list(metas), emb)
        logger.info("FlatChunkIndex: %d개 청크 추가 (총 %d개)", len(ids), len(self._ids) + len(ids))
        return len(ids)

    def remove_doc(self, doc_slug: str) -> int:
        """해당 doc_slug의 청크를 전부 제거하고 저장. 제거된 개수 반환(없으면 0)."""
        self._loaded = False
        self._load()
        if not self._ids:
            return 0
        keep = [i for i, m in enumerate(self._metas) if m.get("doc_slug") != doc_slug]
        removed = len(self._ids) - len(keep)
        if removed == 0:
            return 0
        assert self._emb is not None
        self._save([self._ids[i] for i in keep], [self._docs[i] for i in keep],
                   [self._metas[i] for i in keep], self._emb[keep])
        logger.info("FlatChunkIndex: [%s] 청크 %d개 제거 (잔여 %d개)", doc_slug, removed, len(keep))
        return removed

    # --- 검색 ---
    def query(self, query_text: str, n_results: int = 20, *,
              query_embedding: list[float] | None = None) -> list[ScoredItem]:
        self._load()
        if not self._ids or self._emb is None:
            logger.warning("FlatChunkIndex 비어있음 — ingest 필요")
            return []

        if query_embedding is not None:
            q = np.asarray(query_embedding, dtype=np.float32)
        else:
            q = np.asarray(self.backend.embed([query_text], is_query=True)[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)

        dense_scores = self._emb @ q  # 코사인(정규화됨)
        dense_order = np.argsort(-dense_scores)
        dense_rank = {int(i): r for r, i in enumerate(dense_order)}

        bm25_scores = self._bm25.get_scores(tokenize_ko(query_text, self.config.tokenizer)) if self._bm25 else np.zeros(len(self._ids))
        bm25_order = np.argsort(-bm25_scores)
        bm25_rank = {int(i): r for r, i in enumerate(bm25_order)}

        k = self.config.rrf_k
        max_possible = 2.0 / (k + 1)
        fused = np.zeros(len(self._ids), dtype=np.float32)
        for i in range(len(self._ids)):
            s = 1.0 / (k + dense_rank[i] + 1) + 1.0 / (k + bm25_rank[i] + 1)
            fused[i] = s / max_possible
        top = np.argsort(-fused)[:n_results]

        out = []
        for i in top:
            i = int(i)
            out.append(ScoredItem(
                id=self._ids[i], text=self._docs[i], metadata=self._metas[i],
                score=float(fused[i]), dense_rank=dense_rank[i], bm25_rank=bm25_rank[i],
                dense_similarity=float(dense_scores[i]),
            ))
        return out


def _l2norm(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / (norms + 1e-8)


_FLAT_CACHE: dict[str, FlatChunkIndex] = {}


def get_flat_chunk_index(config: Config, backend: Backend) -> FlatChunkIndex:
    key = f"{config.index_dir}:{backend.backend_id}"
    if key not in _FLAT_CACHE:
        _FLAT_CACHE[key] = FlatChunkIndex(config, backend)
    return _FLAT_CACHE[key]

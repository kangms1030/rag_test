"""Chroma 2-컬렉션(catalog_index/page_index) BM25/dense 하이브리드(RRF) 검색.

test_1차(4컬렉션: catalog/page/visual_chunk/filename)에서 visual_chunk_index/filename_index를
제거했다 — 리트리벌은 카탈로그 문서선정 -> 페이지선정 2단계뿐이고, VLM 청킹이 없다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import chromadb

from . import metrics
from .config import Config
from .models import Backend
from .tokenizer import tokenize_ko

logger = logging.getLogger(__name__)


@dataclass
class ScoredItem:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_similarity: float | None = None


class HybridIndex:
    def __init__(self, name: str, config: Config, backend: Backend):
        self.name = name
        self.config = config
        self.backend = backend
        self.chroma_dir = config.chroma_dir / backend.backend_id
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        #: 전역(where=None) BM25 캐시 — (ids, docs, metas, tokenized_docs, BM25Okapi). 청크 색인 규모(수천건)에서
        #: 질의마다 재토큰화/재구축(F7)을 피하려 1회만 만들고 upsert 시 무효화한다.
        self._bm25_cache: tuple | None = None

    def _reopen(self) -> None:
        """PersistentClient/컬렉션 핸들을 다시 연다(파괴 없이 HNSW 세그먼트 재로드 시도)."""
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})

    def _reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})

    def _with_recovery(self, fn, *, allow_reset: bool = False):
        """HNSW/compactor 오류 시 복구. 읽기(allow_reset=False)는 파괴하지 않고 client 재오픈으로 재시도만 한다.

        test_2의 원본은 어떤 오류에도 컬렉션을 삭제했는데(B6), 이는 읽기 중 일시적 세그먼트
        로드 실패에도 색인 전체(수천 벡터)를 날려 '게이트가 엄격하다'는 가짜 증상으로 위장됐다.
        여기서는 읽기는 재오픈 재시도만, 파괴적 재생성은 명시적(ingest)일 때만 허용한다.
        """
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "hnsw" not in msg and "compactor" not in msg:
                raise
            # 재오픈 후 재시도(최대 2회) — 일시적 로드 실패 대응
            for attempt in range(2):
                logger.warning("%s: Chroma 세그먼트 로드 실패(%s) — client 재오픈 재시도 %d", self.name, e, attempt + 1)
                try:
                    self._reopen()
                    return fn()
                except Exception as e2:
                    e = e2
            if allow_reset:
                logger.error("%s: 재오픈 실패 — 컬렉션 재생성(파괴적). ingest 재실행 필요.", self.name)
                self._reset_collection()
                return fn()
            # 읽기 경로: 파괴하지 않고 오류를 올려 상위에서 처리(색인 보존)
            raise RuntimeError(f"{self.name}: Chroma 세그먼트 로드 실패(색인 보존, 재빌드 필요): {e}") from e

    def count(self) -> int:
        return self._with_recovery(lambda: self.collection.count())

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        embeddings = self.backend.embed(texts, is_query=False)
        self._with_recovery(
            lambda: self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings),
            allow_reset=True,
        )
        self._bm25_cache = None  # 색인 변경 -> BM25 캐시 무효화

    def _get_bm25(self):
        """전역 BM25 캐시 반환 (ids, docs, metas, BM25Okapi). 최초/무효화 시 1회 구축."""
        from rank_bm25 import BM25Okapi

        if self._bm25_cache is None:
            ids, docs, metas = self._all_records()
            tokenized = [tokenize_ko(d, self.config.tokenizer) for d in docs]
            bm25 = BM25Okapi(tokenized) if docs else None
            self._bm25_cache = (ids, docs, metas, bm25)
        return self._bm25_cache

    def _all_records(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        if self.count() == 0:
            return [], [], []
        got = self._with_recovery(lambda: self.collection.get(include=["documents", "metadatas"]))
        return got["ids"], got["documents"], got["metadatas"]

    def all_metadata(self) -> list[dict[str, Any]]:
        return self._all_records()[2]

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        *,
        query_embedding: list[float] | None = None,
    ) -> list[ScoredItem]:
        """query_embedding을 넘기면 재임베딩하지 않고 그대로 재사용한다(질문당 임베딩 1회 원칙).

        where=None(전역)이면 캐시된 BM25를 재사용(F7 대응). where가 있으면 부분집합에 대해 즉석 구축.
        """
        from rank_bm25 import BM25Okapi

        if where is None:
            ids, docs, metas, bm25 = self._get_bm25()
            if not ids:
                logger.warning("%s: 컬렉션이 완전히 비어 있음(count=0) — ingest 재실행 필요할 수 있음", self.name)
                return []
        else:
            ids, docs, metas = self._all_records()
            if not ids:
                return []
            filtered = [(i, d, m) for i, d, m in zip(ids, docs, metas) if all(m.get(k) == v for k, v in where.items())]
            if not filtered:
                return []
            ids, docs, metas = (list(x) for x in zip(*filtered))
            bm25 = BM25Okapi([tokenize_ko(d, self.config.tokenizer) for d in docs])

        if query_embedding is not None:
            q_emb = query_embedding
        else:
            q_emb = self.backend.embed([query_text], is_query=True)[0]
            metrics.record_embed()
        dense_res = self._with_recovery(
            lambda: self.collection.query(
                query_embeddings=[q_emb], n_results=len(ids), where=where or None, include=["distances"]
            )
        )
        dense_order = dense_res["ids"][0]
        dense_rank = {doc_id: rank for rank, doc_id in enumerate(dense_order)}
        dense_similarity = {doc_id: 1.0 - dist for doc_id, dist in zip(dense_order, dense_res["distances"][0])}

        bm25_scores = bm25.get_scores(tokenize_ko(query_text, self.config.tokenizer))
        bm25_order = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -bm25_scores[i])]
        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_order)}

        k = self.config.rrf_k
        max_possible = 2.0 / (k + 1)
        fused: dict[str, float] = {}
        for doc_id in ids:
            s = 0.0
            if doc_id in dense_rank:
                s += 1.0 / (k + dense_rank[doc_id] + 1)
            if doc_id in bm25_rank:
                s += 1.0 / (k + bm25_rank[doc_id] + 1)
            fused[doc_id] = s / max_possible

        id_to_doc = dict(zip(ids, docs))
        id_to_meta = dict(zip(ids, metas))
        ranked = sorted(fused, key=lambda i: -fused[i])[:n_results]
        return [
            ScoredItem(
                id=doc_id,
                text=id_to_doc[doc_id],
                metadata=id_to_meta[doc_id],
                score=fused[doc_id],
                dense_rank=dense_rank.get(doc_id),
                bm25_rank=bm25_rank.get(doc_id),
                dense_similarity=dense_similarity.get(doc_id),
            )
            for doc_id in ranked
        ]


_INDEX_CACHE: dict[str, HybridIndex] = {}


def get_index(name: str, config: Config, backend: Backend) -> HybridIndex:
    key = f"{config.chroma_dir}:{name}:{id(backend)}"
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = HybridIndex(name, config, backend)
    return _INDEX_CACHE[key]

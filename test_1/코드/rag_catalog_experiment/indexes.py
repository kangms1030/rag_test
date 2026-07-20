"""Chroma 4-컬렉션(catalog/page/visual_chunk/filename) + BM25/dense 하이브리드(RRF) 검색.

서버형 벡터DB 대신 로컬 실험용 Chroma PersistentClient를 쓴다. 규모가 작아
(카탈로그 14행, 페이지/청크 수백 개) 매 쿼리마다 컬렉션 전체를 읽어 BM25를
즉석에서 재구성해도 충분히 빠르다 — 별도 BM25 영속화 파일 없이 Chroma를
단일 진실 소스로 둘 수 있어 더 단순하다.

주의: Chroma metadata 값은 str/int/float/bool만 허용된다. bbox 같은 중첩 구조는
호출부에서 x1/y1/x2/y2로 펼쳐서 넘겨야 한다.
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
    #: RRF 점수는 랭크만 반영해 "코퍼스 안에서 상대적으로 1등"이면 절대적 관련성이
    #: 없어도 높게 나올 수 있다 (예: 13개 문서 중 아무 관련 없는 질의도 누군가는 1등).
    #: dense_similarity(코사인 유사도, cosine space 한정)는 절대적 관련성 하한선을
    #: 걸기 위한 보조 신호다. (BM25 원점수는 char-bigram 근사 토크나이저 특성상
    #: 무관한 질의에도 우연한 음절 겹침으로 점수가 나와 하한선으로 쓸 수 없었다.)
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

    def _reset_collection(self) -> None:
        """손상된 컬렉션을 삭제 후 재생성한다. cache/에서 재생성 가능한 색인 데이터이므로 안전하다."""
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})

    def _with_recovery(self, fn):
        """Chroma HNSW 세그먼트 손상(관측된 실제 버그: `Error loading hnsw index`)이 발생하면
        컬렉션을 재생성하고 1회 재시도한다. 그 외 예외는 그대로 전파한다.

        재생성 직후 컬렉션은 비어 있다. page_index/visual_chunk_index는 이후 ask/evaluate가
        디스크 캐시(cache/summaries, cache/chunks)를 통해 lazy하게 다시 채우지만,
        catalog_index/filename_index는 `ingest`가 명시적으로 재실행돼야만 복구된다 — 그 전까지는
        모든 질의가 "정상적으로 실행되지만 결과 0건"이 되어 게이트 거절과 구분이 안 된다
        (실측: catalog_index가 이 경로로 조용히 비어버려 core_16 13문항이 전부 오진단됨).
        """
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "hnsw" not in msg and "compactor" not in msg:
                raise
            logger.error(
                "%s: Chroma 컬렉션 손상 감지(%s) — 재생성 후 재시도. 재생성된 컬렉션은 비어 있다. "
                "catalog_index/filename_index라면 `ingest`를 다시 실행해야 복구된다.",
                self.name,
                e,
            )
            self._reset_collection()
            return fn()

    def count(self) -> int:
        return self._with_recovery(lambda: self.collection.count())

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        embeddings = self.backend.embed(texts, is_query=False)
        metrics.record_llm("embed")
        self._with_recovery(lambda: self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings))

    def _all_records(self) -> tuple[list[str], list[str], list[dict[str, Any]]]:
        if self.count() == 0:
            return [], [], []
        got = self._with_recovery(lambda: self.collection.get(include=["documents", "metadatas"]))
        return got["ids"], got["documents"], got["metadatas"]

    def all_metadata(self) -> list[dict[str, Any]]:
        """전체 레코드의 metadata만 반환 (예: page_index 커버리지 스냅샷용)."""
        return self._all_records()[2]

    def query(
        self, query_text: str, n_results: int = 10, where: dict[str, Any] | None = None
    ) -> list[ScoredItem]:
        from rank_bm25 import BM25Okapi

        ids, docs, metas = self._all_records()
        if not ids:
            if where is None:
                # where 필터 없이 결과가 0건이면 "이 질문과 관련 없다"가 아니라 컬렉션
                # 자체가 완전히 비어 있다는 뜻이다 — 게이트가 거절한 것과 겉으로 구분이 안 되므로
                # 반드시 구분해서 남긴다 (실측: catalog_index가 이 경로로 조용히 비어버림).
                logger.warning("%s: 컬렉션이 완전히 비어 있음(count=0) — 게이트 거절이 아니라 색인 데이터 누락. ingest 재실행 필요할 수 있음", self.name)
            return []

        if where:
            filtered = [(i, d, m) for i, d, m in zip(ids, docs, metas) if all(m.get(k) == v for k, v in where.items())]
            if not filtered:
                return []
            ids, docs, metas = (list(x) for x in zip(*filtered))

        q_emb = self.backend.embed([query_text], is_query=True)[0]
        metrics.record_llm("embed")
        dense_res = self._with_recovery(
            lambda: self.collection.query(
                query_embeddings=[q_emb], n_results=len(ids), where=where or None, include=["distances"]
            )
        )
        dense_order = dense_res["ids"][0]
        dense_rank = {doc_id: rank for rank, doc_id in enumerate(dense_order)}
        # hnsw:space="cosine"이면 Chroma는 cosine DISTANCE(1-유사도)를 반환한다.
        dense_similarity = {doc_id: 1.0 - dist for doc_id, dist in zip(dense_order, dense_res["distances"][0])}

        bm25 = BM25Okapi([tokenize_ko(d, self.config.tokenizer) for d in docs])
        bm25_scores = bm25.get_scores(tokenize_ko(query_text, self.config.tokenizer))
        bm25_order = [ids[i] for i in sorted(range(len(ids)), key=lambda i: -bm25_scores[i])]
        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_order)}

        k = self.config.rrf_k
        # RRF 최댓값(두 신호 모두 1등일 때) = 2/(k+1)로 나눠 0~1 스케일로 정규화한다.
        # 정규화하지 않으면 min_doc_score/doc_score_gap_ratio 같은 절대 임계값이
        # k에 따라 달라지는 아주 작은 원점수(k=60일 때 최댓값 ≈0.033)와 어긋나
        # 완벽한 매칭조차 컷오프에 걸려버린다.
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

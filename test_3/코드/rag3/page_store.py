"""페이지 텍스트/메타의 flat KV 저장소 (small-to-big의 'big' 조회용).

Phase 1에서 청크 검색은 flat_index로 옮겼지만 page_index는 Chroma로 남아 있었다.
리랭크 경로에서 page_index는 벡터검색이 아니라 **page_id로 페이지 텍스트를 조회**하는
KV 용도뿐인데, Chroma HNSW의 크로스프로세스 세그먼트 로드 버그(B6)가 969건 page_index에서도
재발했다. KV 조회에 HNSW가 불필요하므로 단순 json 저장소로 대체해 B6를 원천 제거한다.
(게이트 경로의 page_index 벡터검색은 Chroma를 계속 쓰되, 기본 경로는 이 저장소만 참조한다.)
"""
from __future__ import annotations

import json
from typing import Any

from .config import Config

_cache: dict[str, dict] = {}


def _path(config: Config):
    return config.index_dir / "page_store.json"


def save_page_store(config: Config, ids: list[str], texts: list[str], metas: list[dict]) -> int:
    data = {pid: {"text": txt, "meta": meta} for pid, txt, meta in zip(ids, texts, metas)}
    p = _path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    _cache.pop(str(p), None)
    return len(data)


def load_page_store(config: Config) -> dict[str, dict]:
    p = _path(config)
    key = str(p)
    if key not in _cache:
        _cache[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _cache[key]


def fetch_pages(config: Config, page_ids: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    store = load_page_store(config)
    id2text, id2meta = {}, {}
    for pid in page_ids:
        rec = store.get(pid)
        if rec is not None:
            id2text[pid] = rec.get("text", "")
            id2meta[pid] = rec.get("meta", {})
    return id2text, id2meta

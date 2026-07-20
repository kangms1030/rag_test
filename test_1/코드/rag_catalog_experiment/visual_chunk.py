"""VLM 기반 semantic visual chunking (lazy + 디스크 캐시).

페이지 단위 검색만으로는 표/절차/구성도가 페이지 요약 한 문단에 뭉개져 버려서
세부 근거를 놓치기 쉽다. 이 모듈은 페이지 내부를 의미 단위로 나누고, 각 chunk에
normalized bbox를 부여해 이후 crop 검증이 가능하게 한다.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from . import metrics
from .config import Config
from .imaging import clamp_bbox, is_valid_bbox
from .indexes import HybridIndex
from .models import Backend, LLMJsonError
from .pdf_parse import PageRecord

logger = logging.getLogger(__name__)

_VALID_CHUNK_TYPES = {"title", "paragraph", "table", "figure", "diagram", "procedure", "mixed"}

PROMPT_CHUNK = """이 페이지 이미지를 의미 단위(semantic chunk)로 나눠줘.

반드시 아래 JSON 형식으로만 답해줘. 다른 텍스트는 쓰지 마.
{{
  "chunks": [
    {{
      "chunk_type": "title|paragraph|table|figure|diagram|procedure|mixed",
      "title_or_heading": "이 영역의 제목 또는 소제목 (없으면 빈 문자열)",
      "summary": "이 영역에 무엇이 담겨 있는지 1~2문장 설명",
      "keywords": ["키워드1", "키워드2"],
      "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0
    }}
  ]
}}

규칙:
- bbox 좌표는 페이지 전체 기준 비율(0.0~1.0), 왼쪽위(x1,y1) ~ 오른쪽아래(x2,y2)
- 표, 절차/순서도, 그림, 네트워크 구성도는 하나의 의미 단위로 통째로 유지하고 잘게 쪼개지 마
- 페이지에 의미 단위가 하나뿐이면 chunks 배열에 항목 1개만 넣어도 된다
- 페이지당 chunk는 최대 8개 이내로
- 이미지에서 확인되지 않는 내용을 추측해서 만들지 마
"""


def _iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


def _validate_and_merge(raw_chunks: list[dict], *, max_chunks: int = 12, iou_merge_threshold: float = 0.6) -> list[dict]:
    valid = []
    for c in raw_chunks:
        bbox = {"x1": c.get("x1"), "y1": c.get("y1"), "x2": c.get("x2"), "y2": c.get("y2")}
        if not is_valid_bbox(bbox):
            continue
        b = clamp_bbox(bbox, padding=0.0)
        chunk_type = c.get("chunk_type") if c.get("chunk_type") in _VALID_CHUNK_TYPES else "mixed"
        valid.append(
            {
                "chunk_type": chunk_type,
                "title_or_heading": str(c.get("title_or_heading") or ""),
                "summary": str(c.get("summary") or ""),
                "keywords": [str(k) for k in (c.get("keywords") or [])][:10],
                "bbox": {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2},
            }
        )

    merged: list[dict] = []
    for c in valid:
        merge_target = None
        for m in merged:
            if m["chunk_type"] == c["chunk_type"] and _iou(m["bbox"], c["bbox"]) >= iou_merge_threshold:
                merge_target = m
                break
        if merge_target is None:
            merged.append(c)
        else:
            merge_target["bbox"] = {
                "x1": min(merge_target["bbox"]["x1"], c["bbox"]["x1"]),
                "y1": min(merge_target["bbox"]["y1"], c["bbox"]["y1"]),
                "x2": max(merge_target["bbox"]["x2"], c["bbox"]["x2"]),
                "y2": max(merge_target["bbox"]["y2"], c["bbox"]["y2"]),
            }
            merge_target["summary"] = (merge_target["summary"] + " " + c["summary"]).strip()
            merge_target["keywords"] = list(dict.fromkeys(merge_target["keywords"] + c["keywords"]))[:10]

    return merged[:max_chunks]


def _cache_path(config: Config, backend_id: str, doc_slug: str, page_number: int) -> Path:
    # page_summary.py의 동일 문제 참조: backend_id로 나누지 않으면 mock 실행이 실제 문서
    # 캐시에 mock chunk를 남긴다.
    return config.chunks_dir / backend_id / doc_slug / f"p{page_number:04d}.json"


def get_or_build_chunks(
    page: PageRecord, doc_slug: str, page_summary: str, backend: Backend, config: Config, *, force: bool = False
) -> tuple[list[dict[str, Any]], bool]:
    """(chunks, was_cached) 반환. chunks는 neighbor_context까지 채워진 최종 형태."""
    cache_path = _cache_path(config, backend.backend_id, doc_slug, page.page_number)
    if cache_path.exists() and not force:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics.record_chunk(doc_slug, page.page_number, was_cached=True)
        return data, True

    try:
        raw = backend.chat_vision_json(PROMPT_CHUNK, [Path(page.image_path)])
        raw_chunks = raw.get("chunks", [])
        if not isinstance(raw_chunks, list):
            raw_chunks = []
    except LLMJsonError as e:
        logger.warning("%s p%d visual chunking 실패, 페이지 전체를 단일 chunk로 대체: %s", doc_slug, page.page_number, e)
        raw_chunks = [{"chunk_type": "mixed", "title_or_heading": "", "summary": page_summary, "x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}]

    chunks = _validate_and_merge(raw_chunks)
    if not chunks:
        chunks = [{"chunk_type": "mixed", "title_or_heading": "", "summary": page_summary, "keywords": [], "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}}]

    headings = [c["title_or_heading"] or c["summary"][:40] for c in chunks]
    for i, c in enumerate(chunks):
        siblings = [h for j, h in enumerate(headings) if j != i]
        neighbor = f"페이지 요약: {page_summary}"
        if siblings:
            neighbor += " | 같은 페이지 인접 영역: " + "; ".join(siblings)
        c["neighbor_context"] = neighbor
        c["chunk_id"] = f"{doc_slug}_p{page.page_number:04d}_c{i:02d}"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    metrics.record_chunk(doc_slug, page.page_number, was_cached=False)
    return chunks, False


def ensure_visual_chunks(
    pages: list[PageRecord],
    doc_slug: str,
    page_summaries: dict[int, str],
    backend: Backend,
    config: Config,
    index: HybridIndex,
    *,
    force: bool = False,
) -> int:
    new_count = 0
    for page in pages:
        page_summary = page_summaries.get(page.page_number, "")
        chunks, was_cached = get_or_build_chunks(page, doc_slug, page_summary, backend, config, force=force)
        if not was_cached:
            new_count += 1

        ids, texts, metas = [], [], []
        for c in chunks:
            ids.append(c["chunk_id"])
            search_text = " ".join(filter(None, [c["title_or_heading"], c["summary"], " ".join(c["keywords"]), c["neighbor_context"]]))
            texts.append(search_text)
            metas.append(
                {
                    "document_name": page.document_name,
                    "file_path": page.file_path,
                    "doc_slug": doc_slug,
                    "page_number": page.page_number,
                    "chunk_type": c["chunk_type"],
                    "title_or_heading": c["title_or_heading"],
                    "summary": c["summary"],
                    "bbox_x1": c["bbox"]["x1"],
                    "bbox_y1": c["bbox"]["y1"],
                    "bbox_x2": c["bbox"]["x2"],
                    "bbox_y2": c["bbox"]["y2"],
                    "neighbor_context": c["neighbor_context"],
                    "source_image_path": page.image_path,
                }
            )
        index.upsert(ids, texts, metas)
    if new_count:
        logger.info("%s: visual chunking %d페이지 신규 생성", doc_slug, new_count)
    return new_count

"""VLM 기반 페이지 요약 (검색용, lazy + 디스크 캐시).

`vlm_test_v2.ipynb`의 PROMPT_SUMMARY를 그대로 재사용한다. 페이지 요약은 검색
후보를 좁히기 위한 용도이며 최종 근거로 간주하지 않는다 (근거는 항상 원본
page image / bbox crop 기반).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from . import metrics
from .config import Config
from .indexes import HybridIndex
from .models import Backend
from .pdf_parse import PageRecord

logger = logging.getLogger(__name__)

PROMPT_SUMMARY = """이 페이지를 한 문단으로 요약해줘.
무엇이 담겨 있는지만 설명하면 돼. 상세 내용은 필요 없어.
예) "3계층 네트워크 구조도. Core Router, Distribution Switch, 서버 3대의 연결 관계를 보여줌."
"""


def _cache_path(config: Config, backend_id: str, doc_slug: str, page_number: int) -> Path:
    # backend_id로 분리하지 않으면 mock 백엔드로 돌린 스모크 테스트가 "[MOCK] ..." 요약을
    # 실제 문서 캐시에 그대로 써버려 이후 real 백엔드 실행이 그걸 진짜 요약인 양 재사용한다
    # (실측 사고: Chroma 인덱스만 backend_id로 나뉘어 있고 이 디스크 캐시는 안 나뉘어 있었음).
    return config.summaries_dir / backend_id / doc_slug / f"p{page_number:04d}.json"


def get_or_build_summary(
    page: PageRecord, doc_slug: str, backend: Backend, config: Config, *, force: bool = False
) -> tuple[str, bool]:
    """(summary, was_cached) 반환.

    캐시 히트/미스와 무관하게 이 함수가 호출됐다는 것 자체가 "이 페이지를 요약 대상으로
    결정했다"는 뜻이므로, 캐시 독립적 비용 지표(`summary_pages_required`)는 여기서 기록한다.
    """
    cache_path = _cache_path(config, backend.backend_id, doc_slug, page.page_number)
    if cache_path.exists() and not force:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics.record_summary(doc_slug, page.page_number, was_cached=True)
        return data["summary"], True

    summary = backend.chat_vision_text(PROMPT_SUMMARY, [Path(page.image_path)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "page_number": page.page_number}, f, ensure_ascii=False, indent=2)
    metrics.record_summary(doc_slug, page.page_number, was_cached=False)
    return summary, False


def ensure_page_summaries(
    pages: list[PageRecord],
    doc_slug: str,
    backend: Backend,
    config: Config,
    index: HybridIndex,
    *,
    force: bool = False,
) -> int:
    """주어진 페이지들의 요약을 확보하고 page_index에 upsert. 새로 생성한 개수를 반환.

    Chroma에 이미 있는지 여부로 스킵하지 않는다 — disk 캐시(get_or_build_summary)가
    이미 중복 VLM 호출을 막아주므로, ingest에서 심어둔 pending_vlm placeholder를
    이 함수가 항상 덮어써서 실제 요약으로 승격시킬 수 있어야 한다.
    """
    ids, texts, metas = [], [], []
    new_count = 0
    for page in pages:
        page_id = f"{doc_slug}_p{page.page_number:04d}"
        summary, was_cached = get_or_build_summary(page, doc_slug, backend, config, force=force)
        if not was_cached:
            new_count += 1
        search_text = summary
        if page.extracted_text_if_any.strip():
            search_text += "\n" + page.extracted_text_if_any.strip()[:500]
        ids.append(page_id)
        texts.append(search_text)
        metas.append(
            {
                "document_name": page.document_name,
                "file_path": page.file_path,
                "doc_slug": doc_slug,
                "page_number": page.page_number,
                "page_type": page.page_type,
                "image_path": page.image_path,
                "summary": summary,
                "char_count": page.char_count,
                "pending_vlm": False,
            }
        )
    index.upsert(ids, texts, metas)
    if new_count:
        logger.info("%s: 페이지 요약 %d개 신규 생성", doc_slug, new_count)
    return new_count


def seed_pending_pages(pages: list[PageRecord], doc_slug: str, index: HybridIndex) -> int:
    """ingest 단계: 텍스트가 있는 페이지를 VLM 호출 없이 placeholder로 미리 색인.

    pdfplumber 텍스트를 임시 검색 텍스트로 써서 즉시 검색 가능하게 하고,
    `pending_vlm: True`로 표시해 아직 VLM 요약이 없음을 남긴다. `ensure_page_summaries`가
    같은 id를 upsert하면 실제 요약으로 덮어써진다.
    """
    ids, texts, metas = [], [], []
    for page in pages:
        if page.char_count <= 50:
            continue
        ids.append(f"{doc_slug}_p{page.page_number:04d}")
        texts.append(page.extracted_text_if_any[:1000])
        metas.append(
            {
                "document_name": page.document_name,
                "file_path": page.file_path,
                "doc_slug": doc_slug,
                "page_number": page.page_number,
                "page_type": page.page_type,
                "image_path": page.image_path,
                "summary": "",
                "char_count": page.char_count,
                "pending_vlm": True,
            }
        )
    index.upsert(ids, texts, metas)
    return len(ids)

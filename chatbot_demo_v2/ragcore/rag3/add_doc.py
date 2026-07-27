"""신규 문서 증분 추가/교체/제거 — 전체 재-ingest 없이 기존 인덱스를 확장한다.

전제(사용자 절차): ① Excel 카탈로그(config.catalog_excel_path)에 새 문서 행을 **끝에** 추가
② PDF 파일을 config.documents_dir 아래에 복사 ③ `python -m rag3 add --pdf <파일명>` 실행.

흐름(기존 ingest와 동일 산출 보장):
  1) 카탈로그 로드/매칭으로 대상 행 확정 (매칭 실패 시 근접 후보 안내 후 실패)
  2) MinerU pipeline 파싱 (기존 parse 경로 그대로; source_parsed에 캐시 있으면 재사용)
  3) 파싱 산출을 source_parsed(parsed_v25)로 이전 — ingest가 청크화 소스로 여기만 읽기 때문
  4) figure 페이지 vlm-engine 텍스트화 병합 (Phase 3 품질 단계 재현; 실패해도 add는 계속)
  5) ingest.collect_chunk_records(공유 헬퍼)로 청크 생성 → 신규 청크만 임베딩해 flat 인덱스 append
  6) page_store 병합 + (게이트 경로용) Chroma page/catalog 인덱스 upsert(실패 무해)

같은 문서를 다시 add하면 교체(replace)로 동작한다. 주의: 이미 떠 있는 웹서버/REPL은
인메모리 인덱스 캐시 때문에 재시작해야 새 문서를 본다.
"""
from __future__ import annotations

import glob
import json
import logging
import shutil
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from .catalog import CatalogRow, load_catalog, match_catalog_to_pdfs, save_match_report
from .config import Config
from .flat_index import get_flat_chunk_index
from .ingest import (
    _catalog_prefix_map,
    _load_content_list,
    _load_source_manifest,
    _page_metadata,
    collect_chunk_records,
)
from .models import Backend
from .page_store import load_page_store, save_page_store
from .utils import doc_slug

logger = logging.getLogger(__name__)


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _find_target_row(rows: list[CatalogRow], pdf_name: str) -> CatalogRow:
    """--pdf 인자(파일명 또는 상대경로)에 해당하는 매칭된 카탈로그 행을 찾는다."""
    want = _nfc(Path(pdf_name).name)
    matched = [r for r in rows if r.matched_file_path]
    for r in matched:
        if _nfc(Path(r.matched_file_path).name) == want or _nfc(r.matched_file_path) == _nfc(pdf_name):
            return r
    # 실패: 근접 후보를 알려주고 명확히 실패
    scored = sorted(
        ((fuzz.token_set_ratio(want, _nfc(Path(r.matched_file_path).name)), r) for r in matched),
        key=lambda x: -x[0],
    )
    cands = [f"{r.matched_file_path} ({s:.0f})" for s, r in scored[:3]]
    unmatched = [r.columns.get("title", "") for r in rows if not r.matched_file_path]
    raise FileNotFoundError(
        f"'{pdf_name}'에 해당하는 카탈로그 매칭 행이 없습니다.\n"
        f"  - 근접 후보: {cands}\n"
        f"  - 매칭 안 된 카탈로그 행: {unmatched}\n"
        "  → Excel 카탈로그에 행을 추가하고 PDF를 documents_dir(설정된 문서 폴더)에 복사했는지 확인하세요."
    )


def _relocate_parsed(config: Config, slug: str) -> None:
    """cache/parsed/<slug> → source_parsed/<slug> 이전 + manifest 내 경로 재작성.

    ingest/_load_content_list는 source_parsed만 읽으므로, 신규 파싱 산출을 옮기지 않으면
    page_store에만 실리고 청크가 0개가 되는 함정이 있다(기존 ingest 폴백의 알려진 한계).
    """
    src = config.parsed_dir / slug
    dst = config.source_parsed / slug
    if not src.exists() or src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    manp = dst / "manifest.json"
    if manp.exists():
        man = json.loads(manp.read_text(encoding="utf-8"))
        old, new = str(config.parsed_dir), str(config.source_parsed)
        for p in man.get("pages", []):
            for k in ("page_image_path", "table_crop_path"):
                v = p.get(k) or ""
                if v.startswith(old):
                    p[k] = new + v[len(old):]
        manp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[%s] 파싱 산출 이전: %s -> %s", slug, src, dst)


def _vlm_reparse_new_doc(config: Config, slug: str, abs_pdf: Path) -> dict[str, Any]:
    """Phase 3 품질 단계 재현: figure 페이지만 vlm-engine으로 텍스트화해 캐시에 병합.

    Phase 3 실측 제약을 그대로 따른다 — 문서별 개별 PDF(결합 PDF는 window 스톨),
    MINERU_API_MAX_CONCURRENT_REQUESTS=1(기본 3은 텐서 오류).
    """
    import fitz

    from .vlm_reparse import find_figure_pages, merge_doc, vlm_texts_by_page

    cl = _load_content_list(config, slug)
    if cl is None:
        return {"vlm_pages_merged": 0, "note": "content_list 없음"}
    figs = find_figure_pages(cl[0])  # 0-based page_idx
    if not figs:
        return {"vlm_pages_merged": 0, "note": "figure 페이지 없음(생략)"}

    if shutil.which("mineru") is None:
        raise RuntimeError("mineru CLI를 찾을 수 없음 (intern_chatbot 환경인지 확인)")

    work = config.cache_dir / "vlm_reparse" / slug
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    mini = work / f"{slug}_figs.pdf"
    doc = fitz.open(str(abs_pdf))
    nd = fitz.open()
    for pi in figs:
        nd.insert_pdf(doc, from_page=pi, to_page=pi)
    nd.save(str(mini))
    nd.close()
    doc.close()

    out = work / "out"
    import os
    env = os.environ.copy()
    env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = "1"
    logger.info("[%s] vlm-engine 재파싱: figure %dp (페이지당 ~1-2분)", slug, len(figs))
    subprocess.run(
        ["mineru", "-p", str(mini), "-o", str(out), "-b", "vlm-engine"],
        check=True, env=env, timeout=7200, capture_output=True,
    )
    hits = glob.glob(str(out / mini.stem / "*" / "*_content_list.json"))
    if not hits:
        raise RuntimeError(f"vlm-engine content_list 산출 없음: {out}")
    pt_by_idx = vlm_texts_by_page(json.loads(Path(hits[0]).read_text(encoding="utf-8")))
    page_texts = {figs[k] + 1: pt_by_idx.get(k, "") for k in range(len(figs))}  # 1-based 원본 페이지
    st = merge_doc(config.source_parsed, slug, page_texts)
    return {"vlm_pages_merged": st["pages"], "vlm_chars": st["chars"],
            "vlm_skipped_empty": st["skipped_empty"]}


def _merge_page_store(config: Config, slug: str, page_ids: list[str],
                      texts: list[str], metas: list[dict]) -> int:
    """기존 page_store에서 해당 slug 페이지를 제거 후 신규 레코드 병합 저장."""
    store = dict(load_page_store(config))
    prefix = f"{slug}_p"
    store = {k: v for k, v in store.items() if not k.startswith(prefix)}
    for pid, txt, meta in zip(page_ids, texts, metas):
        store[pid] = {"text": txt, "meta": meta}
    all_ids = list(store.keys())
    return save_page_store(config, all_ids,
                           [store[i]["text"] for i in all_ids],
                           [store[i]["meta"] for i in all_ids])


def _upsert_chroma(config: Config, backend: Backend, row: CatalogRow) -> str:
    """게이트 경로(옵션)용 Chroma catalog 인덱스 동기화. 실패해도 add는 성공(B6 이력).

    page_index(Chroma)는 upsert하지 않는다 — B6(HNSW 크로스프로세스 로드 실패)로 이미
    기본 경로에서 배제됐고(page_store.json이 대체), upsert 시도가 로드 실패→컬렉션
    파괴적 재생성을 유발한다. 게이트 모드를 다시 켤 때는 풀 ingest로 재구축할 것.
    """
    try:
        from .index import get_index
        get_index("catalog_index", config, backend).upsert(
            [row.row_id], [row.catalog_search_text],
            [{
                "document_name": Path(row.matched_file_path).name,
                "file_path": row.matched_file_path,
                "doc_slug": doc_slug(row.matched_file_path),
                "title": row.columns.get("title", ""),
                "theme": row.columns.get("theme", ""),
                "publisher": row.columns.get("publisher", ""),
            }],
        )
        return "ok"
    except Exception as e:
        logger.warning("Chroma(게이트 경로) upsert 실패 — 기본 경로에는 영향 없음: %s", e)
        return f"skipped: {e}"


def _refresh_ingest_summary(config: Config, backend: Backend, note: dict[str, Any]) -> None:
    """ingest_summary.json의 코퍼스 카운트를 실제 저장소 기준으로 갱신(사람용 대시보드)."""
    path = config.output_dir / "ingest_summary.json"
    summary: dict[str, Any] = {}
    if path.exists():
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    store = load_page_store(config)
    slugs = {rec.get("meta", {}).get("doc_slug", "") for rec in store.values()}
    slugs.discard("")
    summary.update({
        "documents_parsed": len(slugs),
        "total_pages": len(store),
        "chunk_index_count": get_flat_chunk_index(config, backend).count(),
        "page_index_count": len(store),
        "last_incremental_update": {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **note,
        },
    })
    summary["total_chunks"] = summary["chunk_index_count"]
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _add_one(config: Config, backend: Backend, rows: list[CatalogRow],
             prefix_map: dict[str, str], pdf_name: str, *,
             run_vlm: bool, force_parse: bool) -> dict[str, Any]:
    t0 = time.monotonic()
    row = _find_target_row(rows, pdf_name)
    rel_path = row.matched_file_path
    slug = doc_slug(rel_path)
    abs_pdf = config.documents_dir / rel_path
    document_name = Path(rel_path).name

    store = load_page_store(config)
    mode = "replace" if any(k.startswith(f"{slug}_p") for k in store) else "add"
    dup_slugs = {rec["meta"].get("doc_slug") for rec in store.values()
                 if rec.get("meta", {}).get("document_name") == document_name} - {slug}
    if dup_slugs:
        logger.warning("[%s] 같은 문서명이 다른 slug(%s)로 이미 색인됨 — 경로가 바뀌었다면 "
                       "기존 것을 --remove로 정리하세요(중복 검색 위험)", document_name, dup_slugs)

    # 1) 파싱 (기존 MinerU pipeline 경로 그대로) — source_parsed 캐시가 있으면 재사용
    manifest_path = config.source_parsed / slug / "manifest.json"
    parsed_fresh = False
    if force_parse or not manifest_path.exists():
        from .parse import get_or_parse_document
        logger.info("[%s] MinerU 파싱 시작 (backend=%s)", document_name, config.mineru_backend)
        doc_info = get_or_parse_document(abs_pdf, rel_path, config, force=force_parse)
        if doc_info.parser_used != "mineru":
            raise RuntimeError(
                f"[{document_name}] MinerU 파싱 실패로 {doc_info.parser_used} 폴백이 사용됨 — "
                "content_list가 없어 청크 색인이 불가합니다. MinerU 설치/오류를 확인 후 재시도하세요.")
        _relocate_parsed(config, slug)
        parsed_fresh = True

    # 2) figure 페이지 vlm-engine 텍스트화(Phase 3) — 실패는 경고 후 계속
    vlm_stats: dict[str, Any] = {"vlm_pages_merged": 0, "note": "생략(--skip-vlm)"}
    if run_vlm:
        try:
            vlm_stats = _vlm_reparse_new_doc(config, slug, abs_pdf)
        except Exception as e:
            logger.warning("[%s] vlm 텍스트화 실패 — 병합 없이 계속(도표 페이지 검색 품질만 영향): %s",
                           document_name, e)
            vlm_stats = {"vlm_pages_merged": 0, "note": f"실패: {e}"}

    # 3) 청크화 (ingest와 동일 소스/포맷)
    doc_info = _load_source_manifest(config, slug)
    if doc_info is None:
        raise RuntimeError(f"[{document_name}] source_parsed에 manifest가 없음: {manifest_path}")
    rec = collect_chunk_records(config, prefix_map, slug, doc_info)
    if rec is None:
        raise RuntimeError(
            f"[{document_name}] MinerU content_list 없음(parser_used={doc_info.parser_used}) — "
            "청크 색인 불가. --force-parse로 MinerU 재파싱을 시도하세요.")
    chunk_ids, chunk_texts, chunk_metas, type_counts = rec

    # 4) 인덱스 증분 갱신 — 신규 청크만 임베딩
    flat = get_flat_chunk_index(config, backend)
    removed = flat.remove_doc(slug)
    added = flat.append(chunk_ids, chunk_texts, chunk_metas)

    page_ids = [f"{slug}_p{p.page_number:04d}" for p in doc_info.pages]
    page_texts = [p.text for p in doc_info.pages]
    page_metas = [_page_metadata(doc_info, p) for p in doc_info.pages]
    _merge_page_store(config, slug, page_ids, page_texts, page_metas)
    chroma_status = _upsert_chroma(config, backend, row)

    return {
        "document_name": document_name,
        "doc_slug": slug,
        "mode": mode,
        "parsed_fresh": parsed_fresh,
        "pages": doc_info.page_count,
        "chunks_added": added,
        "chunks_removed_before": removed,
        "chunk_types": type_counts,
        **vlm_stats,
        "chroma": chroma_status,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }


def add_documents(config: Config, backend: Backend, pdf_names: list[str], *,
                  run_vlm: bool = True, force_parse: bool = False) -> dict[str, Any]:
    """PDF 1개 이상을 기존 인덱스에 증분 추가(또는 교체). 요약 dict 반환."""
    config.ensure_dirs()
    t0 = time.monotonic()
    rows = load_catalog(config)
    report = match_catalog_to_pdfs(rows, config.documents_dir)
    save_match_report(report, config.output_dir / "catalog_match_report.json")
    prefix_map = _catalog_prefix_map(rows)

    results = [
        _add_one(config, backend, rows, prefix_map, name, run_vlm=run_vlm, force_parse=force_parse)
        for name in pdf_names
    ]
    summary = {
        "command": "add",
        "results": results,
        "total_chunks": get_flat_chunk_index(config, backend).count(),
        "total_pages": len(load_page_store(config)),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }
    _refresh_ingest_summary(config, backend,
                            note={"command": "add", "documents": [r["document_name"] for r in results]})
    (config.output_dir / "add_doc_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def remove_document(config: Config, backend: Backend, pdf_or_slug: str) -> dict[str, Any]:
    """문서를 인덱스에서 제거(파싱 캐시는 보존 — 재추가가 저렴). 파일명 또는 doc_slug 지정."""
    config.ensure_dirs()
    store = load_page_store(config)
    known_slugs = {rec.get("meta", {}).get("doc_slug", "") for rec in store.values()}

    slug = pdf_or_slug if pdf_or_slug in known_slugs else None
    if slug is None:
        want = _nfc(Path(pdf_or_slug).name)
        for rec in store.values():
            meta = rec.get("meta", {})
            if _nfc(meta.get("document_name", "")) == want:
                slug = meta.get("doc_slug")
                break
    if slug is None:
        raise FileNotFoundError(
            f"'{pdf_or_slug}'에 해당하는 색인 문서가 없습니다. 색인된 slug: {sorted(known_slugs - {''})}")

    flat = get_flat_chunk_index(config, backend)
    removed_chunks = flat.remove_doc(slug)
    prefix = f"{slug}_p"
    kept = {k: v for k, v in store.items() if not k.startswith(prefix)}
    removed_pages = len(store) - len(kept)
    ids = list(kept.keys())
    save_page_store(config, ids, [kept[i]["text"] for i in ids], [kept[i]["meta"] for i in ids])

    _refresh_ingest_summary(config, backend, note={"command": "remove", "doc_slug": slug})
    return {
        "command": "remove",
        "doc_slug": slug,
        "chunks_removed": removed_chunks,
        "pages_removed": removed_pages,
        "total_chunks": flat.count(),
        "total_pages": len(load_page_store(config)),
        "note": "파싱 캐시(source_parsed)와 Chroma(게이트 경로)는 보존됨",
    }

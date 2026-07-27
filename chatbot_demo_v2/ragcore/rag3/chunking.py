"""MinerU content_list 블록 -> 검색용 청크(small-to-big의 'small').

test_2는 페이지 통짜를 색인해 (F6) embeddinggemma 2K 컨텍스트에서 긴 페이지 꼬리가 잘리고,
반복 서식 스캔 문서에서 페이지 특정이 어려웠다(page_hit@3 55.6%). 여기서는 MinerU가 이미 뽑은
블록(text/table)을 페이지 내 섹션 단위 청크로 쪼갠다. 각 청크는 페이지로 역링크되어(page_number)
답변 시 페이지 전체 텍스트('big')로 승격할 수 있다.

- 노이즈 블록(footer/header/page_number/순수 image)은 제외.
- text 블록: heading(text_level 보유)으로 섹션 경계를 잡고 본문을 target_chars로 묶는다.
- table 블록: 표 1개 = 청크 1개(HTML 보존). 너무 크면 헤더 행을 반복하며 행 분할.
- 카탈로그 프리픽스 주입은 ingest.py가 담당(이 모듈은 카탈로그 비의존).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

_NOISE_TYPES = {"footer", "header", "page_number"}


@dataclass
class Chunk:
    chunk_id: str
    doc_slug: str
    document_name: str
    page_number: int          # 1-based
    block_type: str           # "text" | "table"
    heading_path: str         # "1장 > 1.2 절" 형태(현재 섹션 경로)
    text: str                 # 청크 본문(표는 HTML)
    char_count: int
    table_crop_path: str = ""     # 표 청크: MinerU 표 크롭 이미지
    page_image_path: str = ""     # 페이지 렌더 이미지(vision/근거표시)
    is_scanned: bool = False
    has_table: bool = False
    figure_area_ratio: float = 0.0
    page_type: str = "text"


def _clean(s: str) -> str:
    return (s or "").strip()


def _split_table_rows(table_html: str, max_chars: int) -> list[str]:
    """표 HTML이 max_chars를 넘으면 <tr> 행을 헤더 반복하며 분할. 아니면 원본 1개 반환."""
    if len(table_html) <= max_chars:
        return [table_html]
    rows = re.findall(r"<tr.*?</tr>", table_html, flags=re.DOTALL | re.IGNORECASE)
    if len(rows) <= 2:
        return [table_html]
    header = rows[0]
    body_rows = rows[1:]
    # <table ...> 여는 태그 보존
    m = re.match(r"(.*?<table[^>]*>)", table_html, flags=re.DOTALL | re.IGNORECASE)
    open_tag = m.group(1) if m else "<table>"
    parts: list[str] = []
    cur: list[str] = []
    cur_len = len(open_tag) + len(header)
    for r in body_rows:
        if cur and cur_len + len(r) > max_chars:
            parts.append(open_tag + header + "".join(cur) + "</table>")
            cur = []
            cur_len = len(open_tag) + len(header)
        cur.append(r)
        cur_len += len(r)
    if cur:
        parts.append(open_tag + header + "".join(cur) + "</table>")
    return parts


def _heading_path_str(stack: dict[int, str]) -> str:
    return " > ".join(stack[k] for k in sorted(stack))


def build_chunks(
    content_list: list[dict],
    *,
    doc_slug: str,
    document_name: str,
    page_meta: dict[int, dict],   # page_number(1-based) -> manifest page dict(page_image_path 등)
    images_root: Path,
    config: Config,
) -> list[Chunk]:
    """content_list(단일 문서)를 청크 리스트로 변환."""
    # 페이지별로 블록을 순서 유지하며 그룹화
    by_page: dict[int, list[dict]] = {}
    for it in content_list:
        by_page.setdefault(int(it.get("page_idx", 0)), []).append(it)

    chunks: list[Chunk] = []
    heading_stack: dict[int, str] = {}

    for page_idx in sorted(by_page):
        page_number = page_idx + 1
        pm = page_meta.get(page_number, {})
        page_image_path = pm.get("page_image_path", "")
        is_scanned = bool(pm.get("is_scanned"))
        figure_area_ratio = float(pm.get("figure_area_ratio", 0.0))
        page_type = pm.get("page_type", "text")

        buf: list[str] = []
        buf_len = 0
        c_idx = 0

        def flush_text():
            nonlocal buf, buf_len, c_idx
            body = _clean("\n".join(buf))
            buf = []
            buf_len = 0
            if len(body) < config.chunk_min_chars:
                return
            for piece in _split_by_maxlen(body, config.chunk_max_chars):
                chunks.append(Chunk(
                    chunk_id=f"{doc_slug}_p{page_number:04d}_c{c_idx:03d}",
                    doc_slug=doc_slug, document_name=document_name, page_number=page_number,
                    block_type="text", heading_path=_heading_path_str(heading_stack),
                    text=piece, char_count=len(piece),
                    page_image_path=page_image_path, is_scanned=is_scanned,
                    has_table=False, figure_area_ratio=figure_area_ratio, page_type=page_type,
                ))
                c_idx += 1

        for it in by_page[page_idx]:
            t = it.get("type")
            if t in _NOISE_TYPES or t == "image":
                continue
            if t == "text":
                txt = _clean(it.get("text", ""))
                if not txt:
                    continue
                if "text_level" in it:  # heading
                    flush_text()
                    lvl = int(it.get("text_level", 1))
                    # 더 깊은 레벨 제거 후 현재 레벨 갱신
                    for k in [k for k in heading_stack if k >= lvl]:
                        del heading_stack[k]
                    heading_stack[lvl] = txt
                    buf.append(txt)  # 헤딩 텍스트도 검색되게 청크에 포함
                    buf_len += len(txt)
                else:
                    buf.append(txt)
                    buf_len += len(txt)
                    if buf_len >= config.chunk_target_chars:
                        flush_text()
            elif t == "table":
                flush_text()
                body = _clean(it.get("table_body", ""))
                if not body:
                    continue
                caps = " ".join(it.get("table_caption", []) or [])
                foots = " ".join(it.get("table_footnote", []) or [])
                crop = ""
                if it.get("img_path"):
                    cand = images_root / it["img_path"]
                    if cand.exists():
                        crop = str(cand)
                for piece in _split_table_rows(body, config.chunk_table_split_chars):
                    full = "\n".join(x for x in (caps, piece, foots) if x)
                    chunks.append(Chunk(
                        chunk_id=f"{doc_slug}_p{page_number:04d}_c{c_idx:03d}",
                        doc_slug=doc_slug, document_name=document_name, page_number=page_number,
                        block_type="table", heading_path=_heading_path_str(heading_stack),
                        text=full, char_count=len(full),
                        table_crop_path=crop, page_image_path=page_image_path,
                        is_scanned=is_scanned, has_table=True,
                        figure_area_ratio=figure_area_ratio, page_type="table",
                    ))
                    c_idx += 1
            elif t == "chart":
                content = _clean(str(it.get("content", "")))
                if content:
                    buf.append(content)
                    buf_len += len(content)
        flush_text()

    return chunks


def _split_by_maxlen(text: str, max_chars: int) -> list[str]:
    """max_chars 초과 시 문단/문장 경계 우선으로 분할."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = max(window.rfind("\n"), window.rfind(". "), window.rfind("다. "))
        if cut < max_chars // 2:
            cut = max_chars
        out.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        out.append(remaining)
    return out

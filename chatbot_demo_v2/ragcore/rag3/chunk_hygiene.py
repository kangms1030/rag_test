# -*- coding: utf-8 -*-
"""색인 직전 청크 위생 (chatbot_demo_v2 신설, 2026-07-27 작업 8).

**왜 필요한가** — 서빙 중이던 색인(2,562청크)을 직접 열어 실측한 결과:

- **완전중복 138개(5.4%)** — 같은 원문이 여러 chunk_id 로 색인돼 후보 슬롯(top-20)을 잠식
- **줄 반복 노이즈 65개(2.5%)** — 웹 UI 스크린샷을 OCR 한 페이지에서 브라우저 북마크바와
  887개 학교명 세로 나열이 그대로 들어갔다. 그런데 **이들이 가장 길다**(9,587자×5,
  12,775자×10) → 임베딩·리랭킹 입력을 통째로 차지한다.
- **임베딩 컨텍스트 초과 151개(5.9%)** — embeddinggemma 는 2,048토큰이라 그 뒤가
  **조용히 잘린 채** 색인됐다. 잘렸다는 사실이 어디에도 남지 않았다.

이 모듈은 flat 인덱스 build 직전에 한 번 돌면서 위 셋을 정리하고 리포트를 남긴다.
원본 파싱 캐시(parsed_v25)는 건드리지 않는다 — 색인 입력만 손본다.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: embeddinggemma 컨텍스트 2,048토큰. 한국어는 토큰당 1.2~1.5자, 표 HTML 은 더 짧다.
#: 보수적으로 2,600자를 초과하면 "잘릴 위험"으로 보고 경고한다(제거하지는 않는다).
EMBED_TRUNCATE_WARN_CHARS = 2600

#: 고유 줄 비율이 이 미만이고 줄 수가 충분히 많으면 반복 노이즈로 본다.
NOISE_UNIQUE_RATIO = 0.5
NOISE_MIN_LINES = 20

#: 근사 중복 판정 — 길이가 같고 앞부분이 같으면 사실상 같은 청크로 본다.
#: 완전 해시만으로는 못 잡는다. MinerU 가 같은 페이지를 여러 번 뽑을 때 1~2자씩 달라진다
#: (실측: 8,666자 청크 5개가 chunk_id 만 다른 채 남았다).
NEAR_DUP_PREFIX_CHARS = 400
NEAR_DUP_MIN_CHARS = 1000     # 짧은 청크는 우연히 겹칠 수 있으므로 대상에서 제외

#: 줄바꿈이 아니라 구분자로 이어진 반복도 잡는다("A · A · A · B · B ...").
INLINE_SEPARATORS = (" · ", ", ", " | ")
INLINE_MIN_PARTS = 30


def _body(indexed_text: str) -> str:
    """색인 텍스트에서 카탈로그 프리픽스를 뺀 원문(첫 개행 이후)."""
    nl = indexed_text.find("\n")
    return indexed_text[nl + 1:] if nl != -1 else indexed_text


def _prefix(indexed_text: str) -> str:
    nl = indexed_text.find("\n")
    return indexed_text[: nl + 1] if nl != -1 else ""


def unique_line_ratio(body: str) -> float:
    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    if len(lines) < NOISE_MIN_LINES:
        return 1.0
    return len(set(lines)) / len(lines)


def collapse_repeats(body: str) -> str:
    """연속·반복되는 동일 줄을 1회로 접는다(등장 순서는 보존).

    "학교 / 학교 / 학교 / ... / 대학교 / 대학교" 같은 세로 나열을 의미 손실 없이 줄인다.
    표 HTML(<tr> 반복)은 줄바꿈이 아니라 태그로 이어지므로 이 함수의 영향을 받지 않는다.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(ln)
    return "\n".join(out)


def collapse_inline_repeats(body: str) -> str:
    """구분자로 이어진 반복을 접는다 — "이용명: A 및 A 및 A 및 …" 유형.

    줄바꿈 기반 검출을 빠져나가는 유형이다(한 줄에 전부 이어 붙어 있어 줄 수가 적다).
    표 HTML 은 건드리지 않는다(구분자가 태그라 아래 조건에 걸리지 않는다).
    """
    if "<table" in body or "<td" in body:
        return body
    for sep in INLINE_SEPARATORS:
        parts = body.split(sep)
        if len(parts) < INLINE_MIN_PARTS:
            continue
        if len(set(p.strip() for p in parts)) / len(parts) >= NOISE_UNIQUE_RATIO:
            continue                      # 반복이 아니라 정상 나열
        seen: set[str] = set()
        kept: list[str] = []
        for p in parts:
            s = p.strip()
            if s and s in seen:
                continue
            seen.add(s)
            kept.append(p)
        body = sep.join(kept)
    return body


def near_dup_key(body: str) -> str | None:
    """근사 중복 판정 키. 길이 + 앞부분 해시. 짧은 청크는 대상 아님(None)."""
    if len(body) < NEAR_DUP_MIN_CHARS:
        return None
    head = "".join(body[:NEAR_DUP_PREFIX_CHARS].split())   # 공백 차이 무시
    return f"{len(body)}:{hashlib.md5(head.encode('utf-8')).hexdigest()}"


def sanitize_chunks(
    ids: list[str], texts: list[str], metas: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]], dict[str, Any]]:
    """색인 직전 청크 정리. (ids, texts, metas, report) 반환.

    - 완전중복(원문 해시 동일): 첫 항목만 남긴다
    - 반복 노이즈(고유 줄 비율 낮음): 반복 줄을 접어 압축한다(제거하지 않는다)
    - 임베딩 컨텍스트 초과: **경고만** 남긴다(잘림 사실을 관측 가능하게)
    """
    keep_ids: list[str] = []
    keep_texts: list[str] = []
    keep_metas: list[dict[str, Any]] = []

    seen_hash: dict[str, str] = {}
    seen_near: dict[str, str] = {}
    dropped_dup: list[dict] = []
    dropped_near: list[dict] = []
    compressed: list[dict] = []
    oversize: list[dict] = []

    for cid, text, meta in zip(ids, texts, metas):
        body = _body(text)
        h = hashlib.md5(body.encode("utf-8")).hexdigest()

        if h in seen_hash:
            dropped_dup.append({"chunk_id": cid, "same_as": seen_hash[h],
                                "document_name": meta.get("document_name"),
                                "page_number": meta.get("page_number"), "chars": len(body)})
            continue
        seen_hash[h] = cid

        # 근사 중복 — 완전 해시를 빠져나간 1~2자 차이 사본
        nk = near_dup_key(body)
        if nk is not None:
            if nk in seen_near:
                dropped_near.append({"chunk_id": cid, "same_as": seen_near[nk],
                                     "document_name": meta.get("document_name"),
                                     "page_number": meta.get("page_number"), "chars": len(body)})
                continue
            seen_near[nk] = cid

        before = len(body)
        new_body = body
        if unique_line_ratio(body) < NOISE_UNIQUE_RATIO:
            new_body = collapse_repeats(new_body)
        new_body = collapse_inline_repeats(new_body)
        if len(new_body) < before:
            compressed.append({"chunk_id": cid, "document_name": meta.get("document_name"),
                               "page_number": meta.get("page_number"),
                               "before": before, "after": len(new_body),
                               "unique_line_ratio": round(unique_line_ratio(body), 3)})
            text = _prefix(text) + new_body
            body = new_body
            meta = dict(meta, char_count=len(new_body), hygiene="collapsed_repeats")

        if len(text) > EMBED_TRUNCATE_WARN_CHARS:
            oversize.append({"chunk_id": cid, "document_name": meta.get("document_name"),
                             "page_number": meta.get("page_number"), "chars": len(text),
                             "block_type": meta.get("block_type")})

        keep_ids.append(cid)
        keep_texts.append(text)
        keep_metas.append(meta)

    saved = (sum(d["chars"] for d in dropped_dup) + sum(d["chars"] for d in dropped_near)
             + sum(c["before"] - c["after"] for c in compressed))
    report = {
        "input_chunks": len(ids),
        "output_chunks": len(keep_ids),
        "dropped_duplicates": len(dropped_dup),
        "dropped_near_duplicates": len(dropped_near),
        "compressed_noise": len(compressed),
        "oversize_warned": len(oversize),
        "chars_saved": saved,
        "detail": {
            "duplicates": dropped_dup[:50],
            "near_duplicates": dropped_near[:50],
            "compressed": sorted(compressed, key=lambda c: c["after"] - c["before"])[:50],
            "oversize": sorted(oversize, key=lambda o: -o["chars"])[:50],
        },
    }
    logger.info(
        "청크 위생: %d → %d (중복 -%d, 근사중복 -%d, 반복압축 %d, 초과경고 %d, 절약 %d자)",
        report["input_chunks"], report["output_chunks"], report["dropped_duplicates"],
        report["dropped_near_duplicates"], report["compressed_noise"],
        report["oversize_warned"], report["chars_saved"],
    )
    if oversize:
        logger.warning(
            "임베딩 컨텍스트(%d자) 초과 청크 %d개 — 뒤가 잘린 채 색인됩니다. 최장: %s p%s (%d자)",
            EMBED_TRUNCATE_WARN_CHARS, len(oversize),
            oversize[0]["document_name"], oversize[0]["page_number"], oversize[0]["chars"],
        )
    return keep_ids, keep_texts, keep_metas, report

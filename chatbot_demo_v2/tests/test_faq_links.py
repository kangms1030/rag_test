"""FAQ 근거링크 빌더: source_files 파싱(문서명 승계·페이지 범위)과 문서명 정규화 매칭."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
if str(PKG / "ragcore") not in sys.path:
    sys.path.insert(0, str(PKG / "ragcore"))

from chatbot_demo_v2.scripts.build_faq_doc_links import (  # noqa: E402
    _norm_doc,
    match_doc,
    parse_source_files,
)


def test_parse_single_page():
    assert parse_source_files(["안내서 18쪽"]) == [("안내서", [18])]


def test_parse_page_range():
    assert parse_source_files(["가이드 9~10쪽"]) == [("가이드", [9, 10])]
    assert parse_source_files(["가이드 3-5쪽"]) == [("가이드", [3, 4, 5])]


def test_parse_carries_forward_document_name():
    """'28쪽' 처럼 문서명이 생략되면 직전 문서명을 승계한다(엑셀 표기 관행)."""
    out = parse_source_files(["안내서 7쪽", "28쪽", "가이드 9~10쪽"])
    assert out == [("안내서", [7]), ("안내서", [28]), ("가이드", [9, 10])]


def test_parse_range_is_capped():
    """'1~999쪽' 같은 폭주 범위는 상한으로 잘린다."""
    doc, pages = parse_source_files(["문서 1~999쪽"])[0]
    assert len(pages) <= 12 and pages[0] == 1


def test_parse_doc_without_page():
    assert parse_source_files(["★05__FAQ.docx"]) == [("★05__FAQ.docx", [])]


def test_norm_doc_ignores_symbols_and_extension():
    a = _norm_doc("★23년 학교 유무선 운영·관리 안내서_최종.pdf")
    b = _norm_doc("23년 학교 유무선 운영관리 안내서")
    assert b in a          # 부분포함 매칭이 성립해야 함


def test_match_doc_partial_containment():
    corpus = {
        _norm_doc("★23년 학교 유무선 운영·관리 안내서_최종.pdf"): {
            "document_name": "★23년 학교 유무선 운영·관리 안내서_최종.pdf",
            "doc_slug": "23-b3bffb22", "pages": {18: "23-b3bffb22/pages/p0018.png"},
        }
    }
    hit = match_doc("23년 학교 유무선 운영관리 안내서", corpus)
    assert hit is not None and hit["doc_slug"] == "23-b3bffb22"
    assert match_doc("전혀 다른 문서 이름", corpus) is None


def _corpus(*names_with_pages):
    out = {}
    for name, pages in names_with_pages:
        out[_norm_doc(name)] = {
            "document_name": name, "doc_slug": name[:6],
            "pages": {p: f"slug/pages/p{p:04d}.png" for p in pages},
        }
    return out


def test_fuzzy_match_accepts_abbreviated_title():
    """FAQ 표기가 코퍼스 파일명의 축약이면(격차 충분) 유사매칭으로 연결한다.

    실측 사례: FAQ '스쿨넷서비스 학내망 구축 및 운영·관리 가이드(2018.12)'
              ↔ 코퍼스 '8-1. … 운영관리 **개선을 위한** 가이드.pdf' (78점, 격차 23)
    """
    corpus = _corpus(
        ("8-1. 스쿨넷서비스 학내망 구축 및 운영관리 개선을 위한 가이드.pdf", [1, 2]),
        ("(첨부)_5단계_스쿨넷서비스_제공_가이드.pdf", [1]),
    )
    hit = match_doc("스쿨넷서비스 학내망 구축 및 운영·관리 가이드(2018.12)", corpus)
    assert hit is not None
    assert hit["document_name"].startswith("8-1.")
    assert hit["match_method"].startswith("fuzzy")


def test_fuzzy_match_rejects_when_absent():
    """코퍼스에 정말 없는 문서는 유사매칭으로도 연결하지 않는다(오매칭 방지)."""
    corpus = _corpus(
        ("스마트 단말 관리 시스템[MDM]_교육 매뉴얼_Argos Edu v1.5.pdf", [1]),
        ("통합관제(최고관리자)_20260702.pdf", [1]),
    )
    assert match_doc("★05__FAQ.docx", corpus) is None
    assert match_doc("★홈페이지_쳇봇문의_Q_A.pdf", corpus) is None


def test_fuzzy_match_rejects_ambiguous_margin():
    """1·2위가 비슷하면(격차 부족) 거절한다."""
    corpus = _corpus(
        ("학교 무선망 관리 가이드 A.pdf", [1]),
        ("학교 무선망 관리 가이드 B.pdf", [1]),
    )
    assert match_doc("학교 무선망 관리 가이드", corpus) is None


def test_generated_links_file_shape():
    """생성된 faq_doc_links.json 이 런타임이 기대하는 구조인지."""
    p = PKG / "data" / "faq_doc_links.json"
    if not p.is_file():
        pytest.skip("faq_doc_links.json 미생성 — build_faq_doc_links 실행 필요")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert "links" in data and "stats" in data
    for faq_id, rows in list(data["links"].items())[:5]:
        for row in rows:
            assert {"doc_slug", "document_name", "pages"} <= set(row)
            for pg in row["pages"]:
                assert "page" in pg and "image_rel" in pg

# -*- coding: utf-8 -*-
"""작업 8 회귀 테스트 — 색인 직전 청크 위생.

실측 근거(서빙 색인 2,562청크 직접 분석):
  완전중복 138개(5.4%) · 줄반복 노이즈 65개(2.5%) · 임베딩 컨텍스트 초과 151개(5.9%)
1차 필터 적용 후에도 **1~2자 차이 근사 중복** 12개가 남아 필터를 보강했다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ragcore"))

from rag3.chunk_hygiene import (  # noqa: E402
    collapse_inline_repeats, collapse_repeats, near_dup_key, sanitize_chunks, unique_line_ratio,
)

PREFIX = "문서: X.pdf | 분류: A | p1\n"


def _chunk(body, cid="c1", doc="X.pdf", page=1):
    return cid, PREFIX + body, {"document_name": doc, "page_number": page,
                                "block_type": "text", "char_count": len(body)}


def _run(chunks):
    ids = [c[0] for c in chunks]
    texts = [c[1] for c in chunks]
    metas = [c[2] for c in chunks]
    return sanitize_chunks(ids, texts, metas)


class TestDuplicates:
    def test_완전중복은_첫_항목만_남긴다(self):
        body = "같은 내용입니다.\n" * 5
        ids, texts, metas, rep = _run([_chunk(body, "c1"), _chunk(body, "c2"), _chunk(body, "c3")])
        assert ids == ["c1"]
        assert rep["dropped_duplicates"] == 2

    def test_근사중복도_제거한다(self):
        """MinerU 가 같은 페이지를 여러 번 뽑으면 1~2자씩 달라져 완전 해시를 빠져나간다."""
        base = "가" * 3000
        ids, _, _, rep = _run([_chunk(base, "c1"),
                               _chunk(base[:-1] + "나", "c2"),      # 마지막 1자만 다름
                               _chunk(base[:-2] + "다라", "c3")])
        assert ids == ["c1"], f"근사 중복이 남았다: {ids}"
        assert rep["dropped_near_duplicates"] == 2

    def test_짧은_청크는_근사중복_대상이_아니다(self):
        """짧은 문장은 우연히 길이·앞부분이 겹칠 수 있어 잘못 지우면 안 된다."""
        assert near_dup_key("짧은 문장") is None
        a, b = "AP 관리 화면 설명입니다. " * 8, "AP 관리 화면 설명입니다. " * 8
        ids, _, _, _ = _run([_chunk(a, "c1"), _chunk(b[:-1] + "!", "c2")])
        assert len(ids) == 2, "짧은 청크를 근사중복으로 지우면 안 된다"

    def test_서로_다른_내용은_보존한다(self):
        ids, _, _, rep = _run([_chunk("AP 등록 절차입니다." * 60, "c1"),
                               _chunk("회선 신청 절차입니다." * 60, "c2")])
        assert len(ids) == 2
        assert rep["dropped_duplicates"] == 0
        assert rep["dropped_near_duplicates"] == 0


class TestNoiseCompression:
    def test_줄_반복을_접는다(self):
        """웹 UI 스크린샷 OCR 의 세로 나열(학교/학교/…/대학교) 유형."""
        body = "\n".join(["학교"] * 60 + ["대학교"] * 40)
        assert unique_line_ratio(body) < 0.5
        out = collapse_repeats(body)
        assert out == "학교\n대학교"

    def test_구분자로_이어진_반복도_접는다(self):
        """줄바꿈이 없어 줄 기반 검출을 빠져나가는 유형(실측 8,666자 청크)."""
        body = "이용명: " + " · ".join(["사업자주"] * 50)
        out = collapse_inline_repeats(body)
        assert len(out) < len(body) / 5
        assert "사업자주" in out

    def test_정상_나열은_건드리지_않는다(self):
        body = " · ".join(f"항목{i}" for i in range(40))   # 전부 고유
        assert collapse_inline_repeats(body) == body

    def test_표_HTML_은_건드리지_않는다(self):
        """표는 <tr> 반복이 정상이다 — 접으면 데이터가 사라진다."""
        rows = "".join(f"<tr><td>행{i}</td><td>값</td></tr>" for i in range(40))
        body = f"<table>{rows}</table>"
        assert collapse_inline_repeats(body) == body

    def test_압축시_메타의_char_count_도_갱신된다(self):
        body = "\n".join(["반복줄"] * 60)
        _, texts, metas, rep = _run([_chunk(body, "c1")])
        assert rep["compressed_noise"] == 1
        assert metas[0]["char_count"] == len(texts[0].split("\n", 1)[1])
        assert metas[0]["hygiene"] == "collapsed_repeats"


class TestOversizeWarning:
    def test_임베딩_컨텍스트_초과는_경고만_하고_지우지_않는다(self):
        """잘림 자체를 막을 수는 없지만 **조용히 잘리는 것**은 막아야 한다."""
        ids, _, _, rep = _run([_chunk("나" * 4000, "c1")])
        assert ids == ["c1"], "초과했다고 지우면 안 된다"
        assert rep["oversize_warned"] == 1
        assert rep["detail"]["oversize"][0]["chunk_id"] == "c1"

    def test_정상_길이는_경고하지_않는다(self):
        _, _, _, rep = _run([_chunk("다" * 500, "c1")])
        assert rep["oversize_warned"] == 0


def test_리포트_수치가_일관된다():
    body = "같은 내용\n" * 30
    ids, _, _, rep = _run([_chunk(body, "c1"), _chunk(body, "c2"),
                           _chunk("다른 내용입니다." * 50, "c3")])
    assert rep["input_chunks"] == 3
    assert rep["output_chunks"] == len(ids)
    assert rep["output_chunks"] + rep["dropped_duplicates"] + rep["dropped_near_duplicates"] == 3

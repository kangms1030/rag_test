"""프롬프트 로더: 변수 치환, 주석 제거, mtime 핫리로드, 누락변수 안전성."""

from __future__ import annotations

import os
import time

from chatbot_demo_v2.config.settings import PKG_ROOT
from chatbot_demo_v2.prompts.loader import PromptLoader


def test_render_substitutes_variables(tmp_path):
    (tmp_path / "t.md").write_text("질문: $question / 답: $answer", encoding="utf-8")
    p = PromptLoader(tmp_path)
    assert p.render("t", question="Q1", answer="A1") == "질문: Q1 / 답: A1"


def test_comment_block_is_stripped(tmp_path):
    (tmp_path / "t.md").write_text(
        "<!-- 변수: $a 설명 주석 -->\n본문 $a", encoding="utf-8"
    )
    p = PromptLoader(tmp_path)
    out = p.render("t", a="X")
    assert "주석" not in out
    assert out == "본문 X"


def test_missing_variable_is_safe(tmp_path):
    (tmp_path / "t.md").write_text("$known / $unknown", encoding="utf-8")
    p = PromptLoader(tmp_path)
    out = p.render("t", known="OK")          # unknown 미제공 → 예외 없이 원문 유지
    assert "OK" in out and "$unknown" in out


def test_braces_in_body_are_safe(tmp_path):
    """str.format 이었다면 KeyError 났을 JSON 중괄호가 그대로 살아남아야 함."""
    (tmp_path / "t.md").write_text('예시: {"k": "v"} / $x', encoding="utf-8")
    p = PromptLoader(tmp_path)
    out = p.render("t", x="1")
    assert '{"k": "v"}' in out


def test_hot_reload_on_mtime_change(tmp_path):
    f = tmp_path / "t.md"
    f.write_text("v1 $a", encoding="utf-8")
    p = PromptLoader(tmp_path)
    assert p.render("t", a="X") == "v1 X"

    # 파일 수정(mtime 변경) → 서버 재시작 없이 다음 호출부터 반영
    f.write_text("v2 $a", encoding="utf-8")
    os.utime(f, (time.time() + 1, time.time() + 1))
    assert p.render("t", a="X") == "v2 X"


def test_meta_reports_file_and_mtime(tmp_path):
    (tmp_path / "t.md").write_text("x", encoding="utf-8")
    p = PromptLoader(tmp_path)
    m = p.meta("t")
    assert m["prompt_file"] == "t.md" and m["prompt_mtime"] is not None


def test_shipped_prompts_exist_and_render():
    """실제 배포 프롬프트 5종이 존재하고 변수 치환이 동작해야 한다."""
    p = PromptLoader(PKG_ROOT / "prompts")
    assert "Q" in p.render("contextualize", history="H", question="Q")
    assert "Q" in p.render("composer_rag", question="Q", answer="A",
                           evidence_text="E", history_summary="H")
    assert "ORIG" in p.render("composer_faq", question="Q", original_answer="ORIG",
                              history_summary="H")
    assert "ANS" in p.render("answer_grader", question="Q", answer="ANS")
    assert "CAND" in p.render("clarify", candidates="CAND")

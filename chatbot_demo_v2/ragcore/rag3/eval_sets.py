"""평가셋 JSON(id/question/expected_*) 로더. 한글 콘솔 인코딩 문제를 피하려면 CLI에
질문 텍스트를 직접 넘기지 말고 여기서 id로 조회하거나 UTF-8 파일에서 읽어야 한다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_eval_item(path: str | Path, qid: str) -> dict[str, Any]:
    items = load_eval_set(path)
    for item in items:
        if item.get("id") == qid:
            return item
    raise KeyError(f"평가셋 {path}에서 id={qid!r}를 찾을 수 없음")

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


def doc_slug(rel_path: str) -> str:
    """Korean/특수문자 경로를 ascii-safe한 캐시 디렉터리 이름으로 변환.

    같은 basename을 가진 문서가 다른 폴더에 있을 수 있으므로 경로 해시를 덧붙인다.
    """
    name = Path(rel_path).stem
    ascii_part = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    ascii_part = re.sub(r"[^A-Za-z0-9]+", "-", ascii_part).strip("-").lower()
    if not ascii_part:
        ascii_part = "doc"
    ascii_part = ascii_part[:40]
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:8]
    return f"{ascii_part}-{h}"


def new_run_id(question: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:6]
    return f"{ts}_{h}"

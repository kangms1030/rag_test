"""프롬프트 파일 로더 (핫리로드).

prompts/<name>.md 를 string.Template($변수)로 로드한다. mtime 을 캐시해 파일이 바뀐 경우에만
다시 읽으므로 **서버 재시작 없이** 프롬프트를 수정하면 다음 호출부터 즉시 반영된다.
str.format 이 아니라 Template.safe_substitute 를 쓰므로 본문에 중괄호({})가 있어도 안전하고,
변수가 누락돼도 예외 대신 원문($var)이 남는다.

파일 상단의 `<!-- ... -->` 주석(변수 문서화)은 렌더 시 제거된다.
"""

from __future__ import annotations

import logging
import re
import string
import threading
from pathlib import Path

logger = logging.getLogger("chatbot_demo_v2.prompts")

_COMMENT_RE = re.compile(r"<!--.*?-->\s*", re.DOTALL)


class PromptLoader:
    def __init__(self, prompts_dir: Path):
        self._dir = Path(prompts_dir)
        self._cache: dict[str, tuple[float, string.Template]] = {}
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        return self._dir / f"{name}.md"

    def load(self, name: str) -> string.Template:
        """mtime 이 바뀐 경우에만 파일을 다시 읽는다(핫리로드)."""
        p = self._path(name)
        mtime = p.stat().st_mtime
        with self._lock:
            cached = self._cache.get(name)
            if cached is not None and cached[0] == mtime:
                return cached[1]
            text = _COMMENT_RE.sub("", p.read_text(encoding="utf-8")).strip()
            tmpl = string.Template(text)
            self._cache[name] = (mtime, tmpl)
            logger.debug("프롬프트 로드/갱신: %s.md", name)
            return tmpl

    def load_optional(self, name: str) -> string.Template | None:
        """선택적 프롬프트를 로드한다."""
        try:
            return self.load(name)
        except (FileNotFoundError, OSError):
            logger.debug("선택적 프롬프트 없음: %s.md", name)
            return None

    def render(self, name: str, **variables) -> str:
        """프롬프트를 렌더링한다. 파일이 없으면 예외(설정 오류로 취급)."""
        return self.load(name).safe_substitute(**variables)

    def render_optional(self, name: str, **variables) -> str:
        """선택적 프롬프트를 렌더링한다. 파일이 없으면 빈 문자열을 반환한다."""
        tmpl = self.load_optional(name)
        return tmpl.safe_substitute(**variables) if tmpl is not None else ""

    def meta(self, name: str) -> dict:
        """LangSmith 기록용 — 어떤 프롬프트 파일/버전(mtime)으로 답했는지."""
        try:
            return {"prompt_file": f"{name}.md",
                    "prompt_mtime": round(self._path(name).stat().st_mtime, 1)}
        except OSError:
            return {"prompt_file": f"{name}.md", "prompt_mtime": None}

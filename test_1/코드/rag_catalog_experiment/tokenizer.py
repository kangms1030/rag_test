"""한국어 토크나이저. chromadb 등 무거운 의존성이 없어 check-env/import-qa에서도 import 가능.

기본은 정규식 + 한글 char bigram(형태소 분석기 부재 근사). `tokenizer: "kiwi"`로 설정하면
kiwipiepy 형태소 분석기를 쓰고, 미설치 시 경고 후 char_bigram으로 자동 폴백한다.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_kiwi_instance = None
_kiwi_warned = False


def tokenize_char_bigram(text: str) -> list[str]:
    """한글/영숫자 토큰 + 한글 토큰의 char bigram."""
    tokens = re.findall(r"[가-힣]+|[A-Za-z0-9]+", text.lower())
    out: list[str] = []
    for t in tokens:
        out.append(t)
        if len(t) >= 2 and re.fullmatch(r"[가-힣]+", t):
            out.extend(t[i : i + 2] for i in range(len(t) - 1))
    return out


def _get_kiwi():
    global _kiwi_instance, _kiwi_warned
    if _kiwi_instance is not None:
        return _kiwi_instance
    try:
        from kiwipiepy import Kiwi

        _kiwi_instance = Kiwi()
        return _kiwi_instance
    except ImportError:
        if not _kiwi_warned:
            logger.warning("tokenizer=kiwi 설정이나 kiwipiepy 미설치 -> char_bigram으로 폴백")
            _kiwi_warned = True
        return None


def tokenize_kiwi(text: str) -> list[str]:
    kiwi = _get_kiwi()
    if kiwi is None:
        return tokenize_char_bigram(text)
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    for token in kiwi.tokenize(text):
        if token.tag.startswith(("N", "V", "MAG", "SL", "SN")):
            tokens.append(token.form.lower())
    return tokens


def tokenize_ko(text: str, tokenizer: str = "char_bigram") -> list[str]:
    if tokenizer == "kiwi":
        return tokenize_kiwi(text)
    return tokenize_char_bigram(text)


def kiwi_available() -> bool:
    try:
        import kiwipiepy  # noqa: F401

        return True
    except ImportError:
        return False

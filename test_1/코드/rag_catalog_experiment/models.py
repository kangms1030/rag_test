"""LLM/VLM/embedding backends.

Ollama가 기본 provider다. `--mock` 실행에서는 MockBackend가 대신 쓰이며,
모델을 전혀 호출하지 않고 결정론적인 값을 반환한다 (smoke test / 배선 검증용).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config

logger = logging.getLogger(__name__)


class LLMJsonError(RuntimeError):
    """LLM이 유효한 JSON을 반환하지 못했을 때."""


def extract_json(raw: str) -> dict[str, Any]:
    """```json 펜스 제거 후 첫 '{'부터 중괄호 균형을 맞춰 JSON 객체를 추출한다."""
    text = re.sub(r"```json|```", "", raw).strip()
    start = text.find("{")
    if start == -1:
        raise LLMJsonError(f"JSON 객체를 찾을 수 없음: {raw[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    end = None
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise LLMJsonError(f"중괄호가 닫히지 않음: {raw[:200]!r}")

    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMJsonError(f"JSON 파싱 실패: {e}: {candidate[:300]!r}") from e


@dataclass
class ChatResult:
    text: str
    model_used: str


class Backend:
    """LLM/VLM/embedding 공통 인터페이스."""

    #: Chroma 인덱스 디렉터리를 분리하는 데 쓰는 식별자. 임베딩 벡터의 차원/의미가
    #: backend마다 다르므로(mock=64차원 해시 vs 실제 embeddinggemma=768차원 등),
    #: 같은 디렉터리에 섞어 upsert하면 Chroma가 차원 불일치 에러를 낸다.
    backend_id: str = "unknown"

    def chat_json(self, prompt: str, *, system: str | None = None, retries: int = 1) -> dict[str, Any]:
        raise NotImplementedError

    def chat_text(self, prompt: str, *, system: str | None = None) -> str:
        raise NotImplementedError

    def chat_vision_json(
        self, prompt: str, image_paths: list[Path], *, system: str | None = None, retries: int = 1
    ) -> dict[str, Any]:
        raise NotImplementedError

    def chat_vision_text(self, prompt: str, image_paths: list[Path], *, system: str | None = None) -> str:
        raise NotImplementedError

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        raise NotImplementedError


def _resize_for_vlm(image_path: Path, max_side: int) -> bytes:
    from PIL import Image
    import io

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


class OllamaBackend(Backend):
    def __init__(self, config: Config):
        import ollama

        self._ollama = ollama
        self.config = config
        self.backend_id = f"ollama-{config.embedding_model}"
        self._json_reminder = "\n\n반드시 JSON 객체 하나만 출력해. 다른 텍스트나 설명은 절대 쓰지 마."

    def _chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        try:
            resp = self._ollama.chat(
                model=model,
                messages=messages,
                options={"num_ctx": self.config.ollama_num_ctx, "temperature": 0.0},
                keep_alive=self.config.ollama_keep_alive,
            )
            return resp["message"]["content"]
        except Exception as e:
            if model != self.config.fallback_model:
                logger.warning("모델 %s 호출 실패(%s) -> fallback %s로 재시도", model, e, self.config.fallback_model)
                resp = self._ollama.chat(
                    model=self.config.fallback_model,
                    messages=messages,
                    options={"num_ctx": self.config.ollama_num_ctx, "temperature": 0.0},
                    keep_alive=self.config.ollama_keep_alive,
                )
                return resp["message"]["content"]
            raise

    def chat_text(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat(self.config.llm_model, messages).strip()

    def chat_json(self, prompt: str, *, system: str | None = None, retries: int = 1) -> dict[str, Any]:
        raw = self.chat_text(prompt, system=system)
        try:
            return extract_json(raw)
        except LLMJsonError as e:
            if retries <= 0:
                raise
            logger.warning("JSON 파싱 실패, 재시도: %s", e)
            raw2 = self.chat_text(prompt + self._json_reminder, system=system)
            return extract_json(raw2)

    def chat_vision_json(
        self, prompt: str, image_paths: list[Path], *, system: str | None = None, retries: int = 1
    ) -> dict[str, Any]:
        import base64

        images_b64 = [base64.b64encode(_resize_for_vlm(p, self.config.image_max_side)).decode() for p in image_paths]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images_b64})
        raw = self._chat(self.config.vlm_model, messages).strip()
        try:
            return extract_json(raw)
        except LLMJsonError as e:
            if retries <= 0:
                raise
            logger.warning("VLM JSON 파싱 실패, 재시도: %s", e)
            messages[-1] = {"role": "user", "content": prompt + self._json_reminder, "images": images_b64}
            raw2 = self._chat(self.config.vlm_model, messages).strip()
            return extract_json(raw2)

    def chat_vision_text(self, prompt: str, image_paths: list[Path], *, system: str | None = None) -> str:
        import base64

        images_b64 = [base64.b64encode(_resize_for_vlm(p, self.config.image_max_side)).decode() for p in image_paths]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images_b64})
        return self._chat(self.config.vlm_model, messages).strip()

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = self.config.embed_query_prefix if is_query else self.config.embed_doc_prefix
        out = []
        for t in texts:
            resp = self._ollama.embed(model=self.config.embedding_model, input=prefix + t)
            vec = resp["embeddings"][0] if "embeddings" in resp else resp["embedding"]
            out.append(list(vec))
        return out


class MockBackend(Backend):
    """모델 호출 없이 결정론적인 값을 반환. --mock 플래그로 활성화."""

    backend_id = "mock"

    def __init__(self, config: Config):
        self.config = config

    def chat_text(self, prompt: str, *, system: str | None = None) -> str:
        return "[MOCK] " + prompt[:80]

    def chat_json(self, prompt: str, *, system: str | None = None, retries: int = 1) -> dict[str, Any]:
        if "intent" in prompt and "retrieval_query" in prompt:
            return {
                "intent": "mock_intent",
                "keywords": ["mock"],
                "domain_terms": [],
                "required_data_type": "unknown",
                "candidate_filters": {},
                "retrieval_query": prompt[:100],
                "reason": "mock query analyzer",
            }
        if "is_answer_supported" in prompt:
            return {"is_answer_supported": True, "unsupported_claims": [], "notes": "mock verification"}
        return {"mock": True}

    def chat_vision_json(
        self, prompt: str, image_paths: list[Path], *, system: str | None = None, retries: int = 1
    ) -> dict[str, Any]:
        if "evidence" in prompt:
            return {
                "answer": "[MOCK] 이미지 기반 목업 답변입니다.",
                "evidence": [
                    {
                        "image_index": 1,
                        "description": "[MOCK] 근거 영역",
                        "x1": 0.1,
                        "y1": 0.1,
                        "x2": 0.9,
                        "y2": 0.4,
                    }
                ],
            }
        if "chunk_type" in prompt:
            return {
                "chunks": [
                    {
                        "chunk_type": "mixed",
                        "title_or_heading": "[MOCK] chunk",
                        "summary": "[MOCK] 목업 semantic chunk",
                        "keywords": ["mock"],
                        "bbox": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
                    }
                ]
            }
        return {"mock": True}

    def chat_vision_text(self, prompt: str, image_paths: list[Path], *, system: str | None = None) -> str:
        names = ", ".join(p.stem for p in image_paths)
        return f"[MOCK] {names} 페이지 요약"

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha1(t.encode("utf-8")).digest()
            vec = [((h[i % len(h)] / 255.0) * 2 - 1) for i in range(64)]
            out.append(vec)
        return out


def get_backend(config: Config, *, mock: bool = False) -> Backend:
    if mock:
        return MockBackend(config)
    if config.llm_provider == "ollama":
        return OllamaBackend(config)
    raise ValueError(f"지원하지 않는 provider: {config.llm_provider}")

"""LLM/VLM/embedding backends.

test_1차와 달리 query analyzer/verify용 JSON 호출이 없다 — 문서/페이지 라우팅은
BM25+dense RRF와 결정론적 메타데이터 판정(`retrieve.py`)만으로 하고, 모델은
임베딩 1회 + 최종 답변 1회(텍스트 또는 비전)만 호출한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    text: str
    model_used: str


class Backend:
    """LLM/VLM/embedding 공통 인터페이스."""

    #: Chroma 인덱스 디렉터리를 분리하는 데 쓰는 식별자. 임베딩 벡터의 차원/의미가
    #: backend마다 다르므로 같은 디렉터리에 섞어 upsert하면 Chroma가 차원 불일치 에러를 낸다.
    backend_id: str = "unknown"

    def chat_text(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        raise NotImplementedError

    def chat_vision_text(self, prompt: str, image_paths: list[Path], *, model: str | None = None, system: str | None = None) -> str:
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
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue()


class OllamaBackend(Backend):
    def __init__(self, config: Config):
        import ollama

        self._ollama = ollama
        self.config = config
        self.backend_id = f"ollama-{config.embedding_model}"

    def _chat(self, model: str, messages: list[dict]) -> str:
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

    def chat_text(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._chat(model or self.config.text_answer_model, messages).strip()

    def chat_vision_text(self, prompt: str, image_paths: list[Path], *, model: str | None = None, system: str | None = None) -> str:
        import base64

        images_b64 = [base64.b64encode(_resize_for_vlm(p, self.config.answer_image_max_side)).decode() for p in image_paths]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": images_b64})
        return self._chat(model or self.config.vision_answer_model, messages).strip()

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = self.config.embed_query_prefix if is_query else self.config.embed_doc_prefix
        out = []
        for t in texts:
            resp = self._ollama.embed(model=self.config.embedding_model, input=prefix + t)
            vec = resp["embeddings"][0] if "embeddings" in resp else resp["embedding"]
            out.append(list(vec))
        return out


def get_backend(config: Config) -> Backend:
    if config.llm_provider == "ollama":
        return OllamaBackend(config)
    raise ValueError(f"지원하지 않는 provider: {config.llm_provider}")

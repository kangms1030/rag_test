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

    def _options(self) -> dict:
        opts = {
            "num_ctx": self.config.ollama_num_ctx,
            "temperature": 0.0,
            "seed": getattr(self.config, "ollama_seed", 0),
        }
        npredict = getattr(self.config, "ollama_num_predict", 0)
        if npredict and npredict > 0:
            opts["num_predict"] = npredict  # 출력 토큰 상한(장문 생성 폭주로 인한 지연 방지)
        return opts

    def _call(self, model: str, messages: list[dict]) -> dict:
        """ollama.chat 1회(모델 실패 시 fallback_model로 1회 대체). 전체 resp를 반환한다."""
        try:
            return self._ollama.chat(
                model=model, messages=messages,
                options=self._options(), keep_alive=self.config.ollama_keep_alive,
            )
        except Exception as e:
            if model != self.config.fallback_model:
                logger.warning("모델 %s 호출 실패(%s) -> fallback %s로 재시도", model, e, self.config.fallback_model)
                return self._ollama.chat(
                    model=self.config.fallback_model, messages=messages,
                    options=self._options(), keep_alive=self.config.ollama_keep_alive,
                )
            raise

    @staticmethod
    def _log_meta(resp: dict) -> None:
        import os
        if os.environ.get("RAG_DEBUG_META") != "1":
            return
        try:
            _c = resp["message"]["content"]
            logger.warning("META done_reason=%s eval_count=%s prompt_eval_count=%s content_len=%s tail=%r",
                           resp.get("done_reason"), resp.get("eval_count"), resp.get("prompt_eval_count"),
                           len(_c), _c[-40:])
        except Exception:
            pass

    def _chat(self, model: str, messages: list[dict]) -> str:
        resp = self._call(model, messages)
        self._log_meta(resp)
        # 콜드스타트 whitespace 런어웨이 방어: done_reason=length는 자연 종료(stop)가 아니라
        # 미완/폭주다(콜드 KV에서 답변 도중 공백 토큰을 num_predict까지 폭주 -> 답변 잘림).
        # 동일 호출을 1회 재발행하면 서버 KV 캐시가 웜이라 정상 완성됨을 실측(2026-07-16).
        # temp=0 그리디를 유지하므로 숫자 충실도는 보존된다.
        if resp.get("done_reason") == "length" and getattr(self.config, "ollama_retry_on_length", True):
            from . import metrics
            m = metrics.current()
            budget = getattr(self.config, "ollama_max_length_retries", 2)
            # 문항당 예산 내에서만 재발행(병리적 문맥의 지연 폭발 차단). 예산은 length_retry_count로 계측.
            if m is None or m.length_retry_count < budget:
                resp2 = self._call(model, messages)
                self._log_meta(resp2)
                metrics.record_length_retry()
                # 재발행이 자연 종료(stop)했을 때만 채택. 여전히 length면(진짜 장문/병리) 원답 유지.
                if resp2.get("done_reason") == "stop":
                    resp = resp2
        return resp["message"]["content"]

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

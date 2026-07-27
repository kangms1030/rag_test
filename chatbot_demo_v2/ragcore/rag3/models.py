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

    #: length-retry 시 섭동 온도(첫 호출은 여전히 temp=0). temp=0 그리디가 특정 프롬프트에서
    #: 공백으로 붕괴(빈 응답)하는데, 작은 temperature가 이를 깨고 실제 답변을 복원함(2026-07-21 실측).
    _LENGTH_RETRY_TEMPERATURE = 0.3
    #: length-retry 시 출력 토큰 상한(정당하게 긴 답변이 num_predict에 잘리는 경우 완성시킴).
    #: wifi 문항 실측: 완결에 ~1600토큰 필요(1536 상한에 걸림) -> 재발행은 여유 있게 3072.
    _LENGTH_RETRY_NUM_PREDICT = 3072

    def _options(self, *, overrides: dict | None = None) -> dict:
        opts = {
            "num_ctx": self.config.ollama_num_ctx,
            "temperature": 0.0,
            "seed": getattr(self.config, "ollama_seed", 0),
        }
        npredict = getattr(self.config, "ollama_num_predict", 0)
        if npredict and npredict > 0:
            opts["num_predict"] = npredict  # 출력 토큰 상한(장문 생성 폭주로 인한 지연 방지)
        if overrides:
            opts.update(overrides)
        return opts

    def _call(self, model: str, messages: list[dict], *, options: dict | None = None) -> dict:
        """ollama.chat 1회(모델 실패 시 fallback_model로 1회 대체). 전체 resp를 반환한다."""
        opts = options if options is not None else self._options()
        try:
            return self._ollama.chat(
                model=model, messages=messages,
                options=opts, keep_alive=self.config.ollama_keep_alive,
            )
        except Exception as e:
            if model != self.config.fallback_model:
                logger.warning("모델 %s 호출 실패(%s) -> fallback %s로 재시도", model, e, self.config.fallback_model)
                return self._ollama.chat(
                    model=self.config.fallback_model, messages=messages,
                    options=opts, keep_alive=self.config.ollama_keep_alive,
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
        # done_reason=length는 자연 종료(stop)가 아니라 미완이다. 원인은 두 가지가 겹친다(2026-07-21 실측):
        #  ① temp=0 그리디가 특정 프롬프트에서 답변 도중 공백 토큰으로 붕괴 -> 빈/조각 응답 (wifi 문항).
        #  ② 정당하게 긴 답변(다항목 조치 목록 등)이 num_predict 상한에 그대로 잘림 (~1600토큰 필요).
        # "동일 호출 재발행"은 그리디 결정론이라 ①을 재현할 뿐이라(실측: 재발행 2회 모두 length) 실패했다.
        # 그래서 재발행은 temperature를 소폭 올려(붕괴 탈출) + num_predict를 키워(장문 완성) 섭동한다.
        # 섭동은 이미 length로 망가진 실패 경로에서만 발동하므로 정상(stop) 답변에는 영향이 없다.
        if resp.get("done_reason") == "length" and getattr(self.config, "ollama_retry_on_length", True):
            from . import metrics
            m = metrics.current()
            budget = getattr(self.config, "ollama_max_length_retries", 2)
            # 문항당 예산 내에서만 재발행(병리적 문맥의 지연 폭발 차단). 예산은 length_retry_count로 계측.
            if m is None or m.length_retry_count < budget:
                attempt = 0 if m is None else m.length_retry_count
                base_np = getattr(self.config, "ollama_num_predict", 0) or 0
                perturbed = self._options(overrides={
                    "temperature": self._LENGTH_RETRY_TEMPERATURE,
                    "seed": getattr(self.config, "ollama_seed", 0) + attempt + 1,
                    "num_predict": max(base_np * 2, self._LENGTH_RETRY_NUM_PREDICT),
                })
                resp2 = self._call(model, messages, options=perturbed)
                self._log_meta(resp2)
                metrics.record_length_retry()
                # 재발행이 완성(stop)했거나, 여전히 length여도 원답보다 실질 내용이 더 많으면 채택.
                # (원답이 temp=0 공백 붕괴로 비어있는 경우가 잦아 내용 길이 비교가 유효한 판별이 된다.)
                c1 = (resp.get("message", {}).get("content") or "").strip()
                c2 = (resp2.get("message", {}).get("content") or "").strip()
                if resp2.get("done_reason") == "stop" or len(c2) > len(c1):
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

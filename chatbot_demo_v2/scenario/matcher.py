"""텍스트 정규화 + 유사도 매칭 (LLM 미사용).

두 가지 매처가 있다.

``ScenarioMatcher``
    rapidfuzz ``fuzz.ratio`` — **문자 단위 편집 유사도**. 정확 일치·오탈자 수준만 잡는다.

``SemanticScenarioMatcher`` (2026-07-27 신설, 기본값)
    임베딩 코사인으로 top-k 를 회수한 뒤 BGE 리랭커(크로스인코더)로 재점수한다.
    RAG 검색과 같은 스택을 쓰므로 추가 모델이 없고, FAQ 임베딩은 오프라인으로 미리 계산해
    두므로(``scripts/build_faq_embeddings.py``) 매 턴 비용은 질문 1건 임베딩뿐이다.

    **왜 바꿨나**: 문자 유사도로는 한국어 패러프레이즈가 임계 0.90 을 절대 못 넘어 FAQ 236행이
    사실상 사문화돼 있었다. 골든셋 실측(2026-07-27 기준선): faq_paraphrase 10문항 **라우트
    정확도 0%**. 예) "학교에서 새로운 ap를 설치하고 싶어" ↔ "우리 학교 와이파이는 누가 설치하고
    관리하는 건가요" = 0.4167.

임베딩/리랭커를 못 쓰는 환경(파일 없음·백엔드 실패)에서는 조용히 ``ScenarioMatcher`` 로
폴백하므로 결정론 경로가 죽지 않는다.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz, process

from .models import FaqStore, MatchResult

logger = logging.getLogger("chatbot_demo_v2.matcher")

_WS_RE = re.compile(r"\s+")
# 앞뒤 및 반복 구두점 제거용(문장부호 차이로 인한 오탐 방지)
_TRIM_PUNCT = " \t\r\n?!.,~·…\"'()[]{}"


def normalize_text(s: str) -> str:
    """유사도 비교/정확 일치를 위한 공용 정규화.

    - NFKC 유니코드 정규화
    - 소문자화(영문)
    - 줄바꿈/반복 공백 → 단일 공백
    - 앞뒤 공백·구두점 정리
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = _WS_RE.sub(" ", s)
    s = s.strip().strip(_TRIM_PUNCT).strip()
    return s.lower()


class ScenarioMatcher:
    """자유 입력 질문을 모범 질답과 엄격하게 비교한다.

    절차: 정규화 → 정확 일치(dict) → RapidFuzz fuzz.ratio 상위 2건 →
    best >= threshold AND (best-second) >= margin 이면 채택.
    점수는 0.0~1.0 스케일.
    """

    def __init__(self, faq: FaqStore, threshold: float, margin: float):
        self._faq = faq
        self._threshold = float(threshold)
        self._margin = float(margin)
        self._choices = faq.normalized_choices()  # norm_q -> FaqEntry
        self._keys = list(self._choices.keys())

    def top_candidates(self, normalized_question: str, k: int = 2) -> list[dict]:
        """상위 k개 후보를 [{faq_id, question, score}] 로 반환(clarify 되묻기용)."""
        q = normalized_question or ""
        if not self._keys or not q:
            return []
        results = process.extract(q, self._keys, scorer=fuzz.ratio, limit=k)
        out: list[dict] = []
        for key, raw, _ in results:
            entry = self._choices[key]
            out.append({
                "faq_id": entry.id,
                "question": entry.question,
                "score": round(raw / 100.0, 4),
            })
        return out

    def match(self, normalized_question: str) -> MatchResult:
        q = normalized_question or ""

        # 1) 정확 일치
        exact_entry = self._faq.exact(q)
        if exact_entry is not None:
            return MatchResult(
                decision="exact",
                decision_reason="정규화된 질문이 모범 질문과 완전 일치",
                best_score=1.0,
                second_score=0.0,
                margin_observed=1.0,
                threshold=self._threshold,
                margin_required=self._margin,
                matched_id=exact_entry.id,
                matched_question=exact_entry.question,
                matched_sheet=exact_entry.sheet,
                matched_row=exact_entry.row,
            )

        if not self._keys or not q:
            return MatchResult(
                decision="reject_low_score",
                decision_reason="후보 없음 또는 빈 입력",
                best_score=0.0,
                second_score=0.0,
                margin_observed=0.0,
                threshold=self._threshold,
                margin_required=self._margin,
            )

        # 2) RapidFuzz 상위 2건 (fuzz.ratio: 순서 민감, 엄격)
        results = process.extract(
            q, self._keys, scorer=fuzz.ratio, limit=2
        )
        best_key, best_raw, _ = results[0]
        best = best_raw / 100.0
        second = (results[1][1] / 100.0) if len(results) > 1 else 0.0
        margin_obs = best - second
        best_entry = self._choices[best_key]

        base = dict(
            best_score=best,
            second_score=second,
            margin_observed=margin_obs,
            threshold=self._threshold,
            margin_required=self._margin,
            matched_id=best_entry.id,
            matched_question=best_entry.question,
            matched_sheet=best_entry.sheet,
            matched_row=best_entry.row,
        )

        if best < self._threshold:
            return MatchResult(
                decision="reject_low_score",
                decision_reason=(
                    f"최고 점수 {best:.3f} < 임계값 {self._threshold:.3f}"
                ),
                **base,
            )
        if margin_obs < self._margin:
            return MatchResult(
                decision="reject_ambiguous",
                decision_reason=(
                    f"1~2위 점수 차 {margin_obs:.3f} < 여유 {self._margin:.3f} (애매)"
                ),
                **base,
            )
        return MatchResult(
            decision="accept",
            decision_reason=(
                f"최고 점수 {best:.3f} >= {self._threshold:.3f}, "
                f"여유 {margin_obs:.3f} >= {self._margin:.3f}"
            ),
            **base,
        )


class SemanticScenarioMatcher(ScenarioMatcher):
    """의미 유사도 매처 (2026-07-27 신설). 임베딩 top-k 회수 → 크로스인코더 재점수.

    ``ScenarioMatcher`` 를 상속하므로 정확 일치 경로·``MatchResult`` 스키마·``match()``/
    ``top_candidates()`` 시그니처가 모두 동일하다 → 노드·라우팅·테스트 무변경.

    임베딩/리랭커 준비에 실패하면 부모(문자 유사도)로 자동 폴백한다.
    """

    def __init__(self, faq: FaqStore, threshold: float, margin: float,
                 *, settings=None, topk: int = 10):
        super().__init__(faq, threshold, margin)
        self._topk = int(topk)
        self._settings = settings
        self._ready = False
        self._failed = False
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._questions: list[str] = []
        self._mat = None          # np.ndarray (N, D), L2 정규화
        self._backend = None
        self._reranker = None

    # ---------- 지연 초기화 ----------
    def _ensure(self) -> bool:
        if self._ready:
            return True
        if self._failed:
            return False
        with self._lock:
            if self._ready:
                return True
            if self._failed:
                return False
            try:
                import numpy as np

                from ..rag.adapter_util import prepare_ragcore_imports

                s = self._settings
                emb_path = Path(s.data_dir) / "faq_embeddings.json"
                if not emb_path.is_file():
                    raise FileNotFoundError(
                        "faq_embeddings.json 없음 — scripts/build_faq_embeddings.py 를 먼저 실행")

                prepare_ragcore_imports(s)
                from rag3.config import load_config
                from rag3.models import OllamaBackend
                from rag3.rerank import get_reranker

                payload = json.loads(emb_path.read_text(encoding="utf-8"))
                self._ids = payload["ids"]
                self._questions = payload["questions"]
                mat = np.asarray(payload["vectors"], dtype=np.float32)
                self._mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)

                config = load_config(str(s.ragcore_config))
                self._backend = OllamaBackend(config)      # 임베딩은 항상 로컬
                self._reranker = get_reranker(config)
                self._ready = True
                logger.info("의미 매칭 준비 완료 — FAQ %d건, 리랭커 %s",
                            len(self._ids), config.rerank_model)
                return True
            except Exception as exc:  # noqa: BLE001
                self._failed = True
                logger.warning("의미 매칭 준비 실패(%s: %s) — 문자 유사도로 폴백합니다.",
                               type(exc).__name__, exc)
                return False

    # ---------- 점수 계산 ----------
    def _scored(self, question: str, k: int) -> list[tuple[str, str, float]]:
        """[(faq_id, 질문, 점수)] 를 점수 내림차순 상위 k개로 반환."""
        import numpy as np

        q = np.asarray(self._backend.embed([question], is_query=True)[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        cos = self._mat @ q
        pool = np.argsort(-cos)[: max(self._topk, k)]
        hits = self._reranker.rank(question, [self._questions[i] for i in pool])
        out = [(self._ids[pool[h.index]], self._questions[pool[h.index]], float(h.score))
               for h in hits]
        return out[:k]

    # ---------- 공개 API (부모와 동일 시그니처) ----------
    def top_candidates(self, normalized_question: str, k: int = 2) -> list[dict]:
        q = (normalized_question or "").strip()
        if not q or not self._ensure():
            return super().top_candidates(normalized_question, k)
        try:
            return [{"faq_id": fid, "question": self._faq.get(fid).question if self._faq.get(fid)
                     else text, "score": round(score, 4)}
                    for fid, text, score in self._scored(q, k)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("의미 매칭 후보 조회 실패(%s) — 문자 유사도로 폴백", type(exc).__name__)
            return super().top_candidates(normalized_question, k)

    def match(self, normalized_question: str) -> MatchResult:
        q = (normalized_question or "").strip()

        # 1) 정확 일치는 임베딩보다 우선(비용 0, 확실)
        exact_entry = self._faq.exact(q)
        if exact_entry is not None:
            return MatchResult(
                decision="exact", decision_reason="정규화된 질문이 모범 질문과 완전 일치",
                best_score=1.0, second_score=0.0, margin_observed=1.0,
                threshold=self._threshold, margin_required=self._margin,
                matched_id=exact_entry.id, matched_question=exact_entry.question,
                matched_sheet=exact_entry.sheet, matched_row=exact_entry.row,
            )
        if not q or not self._ensure():
            return super().match(normalized_question)

        try:
            top = self._scored(q, 2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("의미 매칭 실패(%s) — 문자 유사도로 폴백", type(exc).__name__)
            return super().match(normalized_question)
        if not top:
            return super().match(normalized_question)

        best_id, best_q, best = top[0]
        second = top[1][2] if len(top) > 1 else 0.0
        margin_obs = best - second
        entry = self._faq.get(best_id)

        base = dict(
            best_score=best, second_score=second, margin_observed=margin_obs,
            threshold=self._threshold, margin_required=self._margin,
            matched_id=best_id,
            matched_question=entry.question if entry else best_q,
            matched_sheet=entry.sheet if entry else None,
            matched_row=entry.row if entry else None,
        )
        if best < self._threshold:
            return MatchResult(
                decision="reject_low_score",
                decision_reason=f"의미 유사도 {best:.3f} < 임계값 {self._threshold:.3f}",
                **base)
        if margin_obs < self._margin:
            return MatchResult(
                decision="reject_ambiguous",
                decision_reason=f"1~2위 점수 차 {margin_obs:.3f} < 여유 {self._margin:.3f} (애매)",
                **base)
        return MatchResult(
            decision="accept",
            decision_reason=(f"의미 유사도 {best:.3f} >= {self._threshold:.3f}, "
                             f"여유 {margin_obs:.3f} >= {self._margin:.3f}"),
            **base)

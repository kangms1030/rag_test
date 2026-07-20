"""RAG3 발표용 데모 웹서버 (FastAPI, 데모 등급 — 단일 워커/무인증, 외부 공개 금지).

실행:
  cmd /c conda activate intern_chatbot && cd test_3 && python webapp\\server.py --port 8000

- 기동 시 Rag3Engine을 1회 로드하고 딥 워밍업(임베딩+LLM 1토큰)으로 VRAM에 상주시킨다.
- /api/ask 는 동기 처리(질문당 25~150s). GPU 1장이므로 동시 질문은 429로 거절.
- 근거 이미지는 outputs/evidence/<run_id>/에 ascii 파일명으로 복사된 것을 /evidence로 서빙
  (한글/Windows 경로가 URL에 노출되지 않고, 파싱 캐시 전체를 노출하지 않음).
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

_TEST3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TEST3))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("webapp")

_STATIC = Path(__file__).resolve().parent / "static"

# FINAL_REPORT.md (2026-07-16) 확정 수치 — 발표 패널용 정적 벤치마크
BENCHMARK = {
    "basis": "36문항 회귀셋 · FINAL_REPORT.md (2026-07-16)",
    "rows": [
        {"label": "페이지 적중률 @3", "value": "77.4%", "note": "목표 75% 달성 (test_2: 55.6%)"},
        {"label": "문서 적중률 @3", "value": "100%", "note": "test_2: 92.3%"},
        {"label": "환각(근거 없는 생성)", "value": "0건", "note": "못 찾으면 정직하게 회피"},
        {"label": "무관 질문 거절", "value": "100%", "note": "5/5"},
        {"label": "vision 오독률", "value": "0%", "note": "test_2: 86%"},
        {"label": "평균 응답 시간", "value": "63.4초", "note": "최대 148.6초 (한도 180초)"},
        {"label": "평균 모델 호출", "value": "2.97회", "note": "한도 3.5회"},
    ],
}

_ask_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag3 import Rag3Engine
    logger.info("엔진 로딩(리랭커/인덱스)...")
    try:
        engine = Rag3Engine(config_path=app.state.config_path, preload=True)
    except ImportError as e:
        logger.error("[환경 오류] 필수 패키지를 찾을 수 없습니다: %s", e)
        logger.error("  -> intern_chatbot conda 환경에서 실행해야 합니다 (현재 파이썬: %s)", sys.executable)
        logger.error("     cmd /c conda activate intern_chatbot && cd test_3 && python webapp\\server.py --port 8000")
        raise SystemExit(1) from e
    logger.info("딥 워밍업(임베딩 + LLM 상주)...")
    engine.warm_up(deep=True)
    app.state.engine = engine
    app.mount("/evidence", StaticFiles(directory=str(engine.config.evidence_dir)), name="evidence")
    logger.info("준비 완료: http://127.0.0.1:%s", app.state.port)
    yield


app = FastAPI(title="RAG3 데모", lifespan=lifespan)
app.state.config_path = None
app.state.port = 8000


class AskRequest(BaseModel):
    question: str


def _trace_rows(r: dict, config) -> list[dict]:
    """ask_cli._print_trace와 동일한 단계 표를 JSON으로 재구성."""
    m = r.get("metrics", {})
    llm = config.text_answer_model
    rows = [
        {"stage": "S1 질문 임베딩", "model": config.embedding_model,
         "calls": m.get("embed_calls", 0), "note": "질문→벡터"},
        {"stage": "S2 하이브리드 검색", "model": "BM25+dense (모델 아님)", "calls": None,
         "note": "청크 top20 후보"},
        {"stage": "S3 리랭크", "model": config.rerank_model.split("/")[-1],
         "calls": m.get("rerank_calls", 0), "note": "후보 재정렬→상위 페이지 승격"},
    ]
    if m.get("judge_calls"):
        rows.append({"stage": "S3a CRAG 질의재작성", "model": llm, "calls": m["judge_calls"],
                     "note": "검색 실패→질문 재작성 후 재검색"})
    if m.get("text_answer_calls"):
        note = "text 답변 생성" + (f" (+length재발행 {m['length_retry_count']}회)"
                                  if m.get("length_retry_count") else "")
        rows.append({"stage": "S6 답변(text)", "model": llm,
                     "calls": m["text_answer_calls"], "note": note})
    if m.get("vision_answer_calls"):
        rows.append({"stage": "S6 답변(vision)", "model": config.vision_answer_model,
                     "calls": m["vision_answer_calls"], "note": "이미지 전사-후-답변"})
    if m.get("verify_calls"):
        rows.append({"stage": "S7 검증(groundedness)", "model": config.verify_model or llm,
                     "calls": m["verify_calls"], "note": "근거 뒷받침 여부 판정"})
    actions = [h.get("action") for h in r.get("rollback_history", []) if h.get("action")]
    if actions:
        rows.append({"stage": "S8 롤백", "model": "결정론 (모델 아님)", "calls": None,
                     "note": ", ".join(actions)})
    return rows


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"warm": True, **app.state.engine.health()}


@app.get("/api/stats")
def stats():
    return {"benchmark": BENCHMARK, "corpus": app.state.engine.health()}


@app.post("/api/warmup")
def warmup():
    if not _ask_lock.acquire(blocking=False):
        raise HTTPException(429, "질문 처리 중에는 워밍업할 수 없습니다")
    try:
        t0 = time.time()
        app.state.engine.warm_up(deep=True)
        return {"ok": True, "elapsed_seconds": round(time.time() - t0, 1)}
    finally:
        _ask_lock.release()


@app.post("/api/ask")
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(400, "질문이 비어 있습니다")
    if not _ask_lock.acquire(blocking=False):
        raise HTTPException(429, "이미 다른 질문을 처리 중입니다. 잠시 후 다시 시도하세요.")
    try:
        engine = app.state.engine
        t0 = time.time()
        r = engine.ask(question, save_evidence=True)
        elapsed = time.time() - t0

        # 근거 이미지: export된 ascii 파일 → /evidence URL (rank 기준 페이지/표크롭 매핑)
        run_id = r.get("run_id", "")
        url_by_rank: dict[int, dict[str, str]] = {}
        for f in r.get("evidence_files", []):
            url_by_rank.setdefault(f["rank"], {})[f["kind"]] = f"/evidence/{run_id}/{Path(f['file']).name}"

        evidence = []
        for rank, (ev, page) in enumerate(zip(r.get("evidence", []), r.get("selected_pages", [])), start=1):
            urls = url_by_rank.get(rank, {})
            evidence.append({
                "document_name": ev.get("document_name"),
                "page_number": ev.get("page_number"),
                "page_score": page.get("page_score"),
                "image_url": urls.get("page"),
                "table_crop_url": urls.get("table_crop"),
            })

        v = r.get("verification") or {}
        flags = []
        if v.get("unsupported_claims"):
            flags.append(f"미지원숫자={v['unsupported_claims']}")
        if v.get("transcription_ocr_mismatch"):
            flags.append("전사-OCR불일치")
        if v.get("abstain"):
            flags.append("회피")

        m = r.get("metrics", {})
        llm_total = sum(m.get(k, 0) for k in
                        ("text_answer_calls", "vision_answer_calls", "judge_calls", "verify_calls"))
        return {
            "run_id": run_id,
            "question": question,
            "final_answer": r.get("final_answer", ""),
            "answer_path": r.get("answer_path"),
            "confidence": r.get("confidence"),
            "rerank_top_score": r.get("rerank_top_score"),
            "evidence": evidence,
            "verification": {"ok": not flags, "flags": flags} if v else None,
            "trace": {
                "rows": _trace_rows(r, engine.config),
                "llm_total": llm_total,
                "embed": m.get("embed_calls", 0),
                "rerank": m.get("rerank_calls", 0),
                "length_retry": m.get("length_retry_count", 0),
                "timings": m.get("timings_seconds", {}),
            },
            "elapsed_seconds": round(elapsed, 1),
        }
    finally:
        _ask_lock.release()


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description="RAG3 데모 웹서버")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=None, help="config.yaml 경로(기본: rag3/config.yaml)")
    args = ap.parse_args()
    app.state.config_path = args.config
    app.state.port = args.port
    uvicorn.run(app, host="127.0.0.1", port=args.port)

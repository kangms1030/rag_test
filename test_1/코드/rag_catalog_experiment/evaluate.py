"""외부 평가셋(human|synthetic|sample_qa)을 로드해 파이프라인을 돌리고 지표를 집계.

세 source는 절대 섞어서 결론 내지 않는다 (aggregate가 human/synthetic/sample_qa/all로
따로 나온다):
- human만 `expected_documents`가 신뢰 가능해 doc_recall/doc_mrr/page_recall을 실측할 수 있다.
- sample_qa는 `expected_documents`가 항상 비어 있어 해당 지표가 전부 None(집계 시 분모에서
  제외)이 된다 — 대신 answer_token_f1/키워드 recall/거절률/비용 지표로 답변 품질을 본다.
- synthetic은 LLM이 만든 질문이라 human과 절대 합쳐 doc_recall 등을 계산하지 않는다.
"""
from __future__ import annotations

import glob
import json
import logging
import time
from pathlib import Path
from typing import Any

from .answer import build_page_evidence, generate_answer, verify_answer
from .config import Config
from .eval_sets import load_eval_set  # re-export: evaluate.py 기존 임포트 경로 호환
from .metrics import RunMetrics, run_metrics
from .models import Backend
from .retrieval import run_retrieval
from .tokenizer import tokenize_ko
from .utils import new_run_id

logger = logging.getLogger(__name__)

#: 비교 실험(compare)이 그대로 평균 낼 수 있는 수치 필드. per-item row에 평탄화해서 담는다.
_METRIC_AVG_FIELDS = (
    "elapsed_seconds_total",
    "selected_doc_count",
    "selected_page_count",
    "selected_chunk_count",
    "evidence_count",
    "summary_pages_required",
    "chunk_pages_required",
    "summary_calls",
    "chunk_calls",
    "answer_vlm_calls",
    "verify_llm_calls",
    "query_analyzer_calls",
    "embed_calls",
)


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens, gold_tokens = tokenize_ko(pred), tokenize_ko(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    overlap = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def _keyword_recall(answer_text: str, keywords: list[str]) -> float | None:
    if not keywords:
        return None
    answer_lower = answer_text.lower()
    hits = sum(1 for k in keywords if k.lower() in answer_lower)
    return hits / len(keywords)


def _find_doc_rel_path(document_name: str, documents_dir: Path) -> str | None:
    # document_name은 파일명이지 glob 패턴이 아니다 — "[MDM]" 같은 대괄호가 든 실제
    # 파일명이 있어 rglob에 그대로 넘기면 문자 클래스로 해석돼 매칭이 조용히 실패한다.
    for p in documents_dir.rglob(glob.escape(document_name)):
        return str(p.relative_to(documents_dir))
    return None


def _is_scanned_target(expected_documents: list[str], config: Config) -> bool | None:
    """expected_documents 중 하나라도 스캔 문서면 True — no_catalog의 스캔 문서 사각지대를
    질문 단위로 층화해서 보기 위한 보조 필드. expected_documents가 없으면 None(측정 불가)."""
    if not expected_documents:
        return None
    from .pdf_parse import get_or_parse_document

    resolved_any = False
    for name in expected_documents:
        rel = _find_doc_rel_path(name, config.documents_dir)
        if rel is None:
            continue
        resolved_any = True
        try:
            info = get_or_parse_document(config.documents_dir / rel, rel, config)
        except Exception:
            continue
        if info.is_scanned:
            return True
    return False if resolved_any else None


def evaluate_one(
    question_item: dict, config: Config, backend: Backend, *, retrieval_mode: str = "catalog", depth: str = "answer"
) -> dict[str, Any]:
    """depth="docs"|"pages"는 VLM 답변 생성을 건너뛴다 — doc_recall/page_recall만 저렴하게
    스윕하고 싶을 때(예: thresholds 리포트) 쓴다. 이 경우 answer_token_f1/abstention 계열
    지표는 "답변을 시도하지 않음"을 뜻하며 실제 거절/품질과 혼동하면 안 된다(row의 depth로 구분)."""
    question = question_item["question"]
    run_metric = RunMetrics(mode=retrieval_mode, depth=depth)

    t0 = time.monotonic()
    with run_metrics(run_metric):
        retrieval = run_retrieval(question, config, backend, mode=retrieval_mode, depth=depth)
        if depth == "answer":
            answer_result = generate_answer(question, retrieval, backend, config)
            run_id = new_run_id(question)
            page_evidence = build_page_evidence(answer_result["raw_evidence"], answer_result["images_used"], retrieval.selected_documents, run_id, config)
            verification = verify_answer(question, answer_result["final_answer"], page_evidence, backend, config)
        else:
            answer_result = {"final_answer": "", "raw_evidence": [], "images_used": [], "skip_reason": f"depth={depth}: 답변 생성 생략"}
            page_evidence = []
            verification = {
                "is_answer_supported": None,
                "unsupported_claims": [],
                "numeric_claims": [],
                "numeric_verification_notes": "",
                "notes": f"depth={depth}: 검증 생략",
            }
    elapsed = time.monotonic() - t0

    expected_docs = question_item.get("expected_documents", [])
    expected_docs_set = set(expected_docs)
    selected_doc_names = [d["document_name"] for d in retrieval.selected_documents]

    if expected_docs_set:
        doc_hit: bool | None = bool(expected_docs_set & set(selected_doc_names))
        doc_rank = next((i + 1 for i, name in enumerate(selected_doc_names) if name in expected_docs_set), None)
        doc_mrr: float | None = (1.0 / doc_rank) if doc_rank else 0.0
    else:
        doc_hit, doc_mrr = None, None

    expected_pages = set(question_item.get("expected_pages", []))
    if expected_pages and expected_docs_set:
        selected_page_nums = {p["page_number"] for p in retrieval.selected_pages if p["document_name"] in expected_docs_set}
        page_hit: bool | None = bool(expected_pages & selected_page_nums)
    else:
        page_hit = None

    # skip_reason이 있으면 VLM을 부르지 않고 조기 종료한 것 — "거절/미답변"의 결정적 신호.
    # 이걸 answer_token_f1에 그대로 태우면 거절 능력이 답변 품질 저하로 잘못 집계된다(설계상 분리).
    abstained = bool(answer_result.get("skip_reason"))
    expected_answer = question_item.get("expected_answer", "")
    f1 = None if abstained or not expected_answer.strip() else _token_f1(answer_result["final_answer"], expected_answer)
    keyword_recall = None if abstained else _keyword_recall(answer_result["final_answer"], question_item.get("expected_answer_keywords", []))

    question_type = question_item.get("question_type", "")
    rejection_correct = abstained if question_type == "irrelevant" else None

    row: dict[str, Any] = {
        "id": question_item["id"],
        "question": question,
        "source": question_item["source"],
        "question_type": question_type,
        "category": question_item.get("category", ""),
        "target_is_scanned": _is_scanned_target(expected_docs, config),
        "doc_recall_hit": doc_hit,
        "doc_mrr": doc_mrr,
        "page_recall_hit": page_hit,
        "answer_token_f1": f1,
        "expected_answer_keyword_recall": keyword_recall,
        "abstained": abstained,
        "rejection_correct": rejection_correct,
        "selected_documents": selected_doc_names,
        "final_answer": answer_result["final_answer"],
        "is_answer_supported": verification.get("is_answer_supported"),
        "evidence_count": len(page_evidence),
        "selected_doc_count": len(retrieval.selected_documents),
        "selected_page_count": len(retrieval.selected_pages),
        "selected_chunk_count": len(retrieval.selected_chunks),
    }
    row.update(run_metric.to_dict())
    row["elapsed_seconds_total"] = round(elapsed, 3)
    return row


def _avg(rows: list[dict], field: str) -> float | None:
    vals = [r[field] for r in rows if r.get(field) is not None]
    return (sum(vals) / len(vals)) if vals else None


def _rate(rows: list[dict], field: str) -> tuple[float | None, int]:
    """field가 bool인 행만 골라 True 비율과 분모(측정 가능했던 행 수)를 함께 반환."""
    vals = [r[field] for r in rows if r.get(field) is not None]
    return ((sum(vals) / len(vals)) if vals else None), len(vals)


def aggregate_rows(rows: list[dict], *, stratify: bool = True) -> dict[str, Any]:
    if not rows:
        return {"count": 0}

    n = len(rows)
    doc_recall_rate, doc_recall_n = _rate(rows, "doc_recall_hit")
    page_recall_rate, page_recall_n = _rate(rows, "page_recall_hit")
    abstention_rate, _ = _rate(rows, "abstained")
    rejection_rate, rejection_n = _rate(rows, "rejection_correct")

    result: dict[str, Any] = {
        "count": n,
        "doc_recall_at_k": doc_recall_rate,
        "doc_recall_n": doc_recall_n,  # expected_documents가 있어 실제로 측정된 항목 수
        "doc_mrr": _avg(rows, "doc_mrr"),
        "page_recall_at_k": page_recall_rate,
        "page_recall_n": page_recall_n,
        "answer_token_f1_avg": _avg(rows, "answer_token_f1"),
        "expected_answer_keyword_recall_avg": _avg(rows, "expected_answer_keyword_recall"),
        "abstention_rate": abstention_rate,
        "rejection_accuracy": rejection_rate,  # question_type=="irrelevant"인 항목만의 정답 거절률
        "rejection_n": rejection_n,
    }
    for f in _METRIC_AVG_FIELDS:
        result[f"avg_{f}"] = _avg(rows, f)

    if stratify:
        scanned_rows = [r for r in rows if r.get("target_is_scanned") is True]
        text_rows = [r for r in rows if r.get("target_is_scanned") is False]
        if scanned_rows or text_rows:
            result["by_target_type"] = {
                "scanned": aggregate_rows(scanned_rows, stratify=False),
                "text": aggregate_rows(text_rows, stratify=False),
            }
    return result


def run_evaluation(
    eval_items: list[dict],
    config: Config,
    backend: Backend,
    *,
    limit: int | None = None,
    retrieval_mode: str = "catalog",
    depth: str = "answer",
) -> dict[str, Any]:
    if limit:
        eval_items = eval_items[:limit]

    rows = []
    for i, item in enumerate(eval_items, start=1):
        logger.info("[%d/%d] 평가 중: %s (%s)", i, len(eval_items), item["question"], item["source"])
        try:
            rows.append(evaluate_one(item, config, backend, retrieval_mode=retrieval_mode, depth=depth))
        except Exception as e:
            logger.error("평가 실패 (id=%s): %s", item.get("id"), e)
            rows.append({"id": item.get("id"), "question": item["question"], "source": item["source"], "error": str(e)})

    ok_rows = [r for r in rows if "error" not in r]
    buckets = {src: [r for r in ok_rows if r["source"] == src] for src in ("human", "synthetic", "sample_qa")}

    return {
        "retrieval_mode": retrieval_mode,
        "depth": depth,
        "per_item": rows,
        "aggregate": {
            "human": aggregate_rows(buckets["human"]),
            "synthetic": aggregate_rows(buckets["synthetic"]),
            "sample_qa": aggregate_rows(buckets["sample_qa"]),
            "all": aggregate_rows(ok_rows),
        },
        "errors": [r for r in rows if "error" in r],
    }


def save_evaluation(result: dict[str, Any], config: Config) -> Path:
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.output_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"evaluation_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    logger.info("평가 결과 저장: %s", path)
    return path

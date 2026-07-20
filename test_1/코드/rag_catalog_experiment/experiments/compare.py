"""동일 평가셋으로 retrieval_mode(catalog/no_catalog/filename_only)를 나란히 돌려 비교.

비용 지표는 두 가지를 함께 낸다:
- `avg_vlm_calls` — 실측 호출 수. 캐시 온도(실행 순서)에 따라 달라져 재현 불가능하다.
- `avg_vlm_pages_required` — 중복 제거된 (문서,페이지) 결정 수. 캐시와 무관해 이 값으로
  비용 결론을 낸다 (`retrieval.py`/`metrics.py` 모듈 docstring 참조).

`doc_recall_at_k`/`page_recall_at_k`는 `expected_documents`/`expected_pages`가 있는 항목에만
의미가 있다 — evaluate.py가 이미 null 처리해 두므로 여기서는 그대로 통과시킨다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config
from ..evaluate import aggregate_rows, run_evaluation
from ..models import Backend

logger = logging.getLogger(__name__)

_BUCKET_SOURCES = ("human", "synthetic", "sample_qa", "all")


def _summarize(agg: dict[str, Any], rows: list[dict[str, Any]], *, mode: str, group: str) -> dict[str, Any]:
    n = agg.get("count", 0)
    answered = sum(1 for r in rows if not r.get("abstained"))
    vlm_calls_avg = None
    if agg.get("avg_summary_calls") is not None:
        vlm_calls_avg = (agg.get("avg_summary_calls") or 0) + (agg.get("avg_chunk_calls") or 0) + (agg.get("avg_answer_vlm_calls") or 0)
    vlm_pages_required_avg = None
    if agg.get("avg_summary_pages_required") is not None:
        vlm_pages_required_avg = (agg.get("avg_summary_pages_required") or 0) + (agg.get("avg_chunk_pages_required") or 0)

    return {
        "retrieval_mode": mode,
        "group": group,
        "total_questions": n,
        "answered_count": answered,
        "no_answer_count": n - answered,
        "avg_elapsed_seconds": agg.get("avg_elapsed_seconds_total"),
        "avg_vlm_calls": vlm_calls_avg,
        "avg_vlm_pages_required": vlm_pages_required_avg,
        "avg_selected_doc_count": agg.get("avg_selected_doc_count"),
        "avg_selected_page_count": agg.get("avg_selected_page_count"),
        "avg_selected_chunk_count": agg.get("avg_selected_chunk_count"),
        "answer_token_f1_avg": agg.get("answer_token_f1_avg"),
        "expected_answer_keyword_recall_avg": agg.get("expected_answer_keyword_recall_avg"),
        "doc_recall_at_k": agg.get("doc_recall_at_k"),
        "doc_recall_n": agg.get("doc_recall_n"),
        "doc_mrr": agg.get("doc_mrr"),
        "page_recall_at_k": agg.get("page_recall_at_k"),
        "page_recall_n": agg.get("page_recall_n"),
        "abstention_rate": agg.get("abstention_rate"),
        "rejection_accuracy": agg.get("rejection_accuracy"),
        "rejection_n": agg.get("rejection_n"),
        "by_target_type": agg.get("by_target_type"),
    }


def _bucket_summaries(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ok_rows = [r for r in result["per_item"] if "error" not in r]
    buckets = {src: [r for r in ok_rows if r["source"] == src] for src in ("human", "synthetic", "sample_qa")}
    buckets["all"] = ok_rows
    return {b: _summarize(result["aggregate"][b], rows, mode=result["retrieval_mode"], group=b) for b, rows in buckets.items()}


def _stratified_summaries(result: dict[str, Any], stratify_field: str) -> dict[str, dict[str, Any]]:
    ok_rows = [r for r in result["per_item"] if "error" not in r]
    groups: dict[str, list[dict]] = {}
    for r in ok_rows:
        key = r.get(stratify_field) or "(unknown)"
        groups.setdefault(key, []).append(r)
    return {key: _summarize(aggregate_rows(rows), rows, mode=result["retrieval_mode"], group=key) for key, rows in groups.items()}


def run_compare(
    eval_items: list[dict],
    config: Config,
    backend: Backend,
    *,
    modes: list[str],
    limit: int | None = None,
    depth: str = "answer",
    stratify: str | None = None,
) -> dict[str, Any]:
    per_mode_results: dict[str, dict[str, Any]] = {}
    for mode in modes:
        logger.info("=== compare: retrieval_mode=%s (%d문항, depth=%s) ===", mode, len(eval_items[:limit] if limit else eval_items), depth)
        per_mode_results[mode] = run_evaluation(eval_items, config, backend, limit=limit, retrieval_mode=mode, depth=depth)

    summary = {mode: _bucket_summaries(result) for mode, result in per_mode_results.items()}

    stratified = None
    if stratify:
        stratified = {mode: _stratified_summaries(result, stratify) for mode, result in per_mode_results.items()}

    return {
        "modes": modes,
        "depth": depth,
        "eval_item_count": len(eval_items[:limit] if limit else eval_items),
        "summary": summary,
        "stratified_by": stratify,
        "stratified": stratified,
        "full_results": per_mode_results,
    }


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_compare_markdown(compare_result: dict[str, Any]) -> str:
    lines = [
        "# 비교 실험 결과 (catalog vs no_catalog vs filename_only)",
        "",
        f"- 모드: {', '.join(compare_result['modes'])}",
        f"- depth: {compare_result['depth']}",
        f"- 평가 문항 수: {compare_result['eval_item_count']}",
        "",
        "`avg_vlm_pages_required`가 비용 비교의 본문 지표다(캐시 독립적, 중복 제거된 페이지 수). "
        "`avg_vlm_calls`는 실행 순서/캐시 온도에 따라 달라지는 참고값이다.",
        "",
    ]

    cols = [
        ("total_questions", "문항"),
        ("answered_count", "답변"),
        ("no_answer_count", "미답변"),
        ("doc_recall_at_k", "doc_recall@k"),
        ("doc_mrr", "doc_mrr"),
        ("page_recall_at_k", "page_recall@k"),
        ("answer_token_f1_avg", "token_f1"),
        ("expected_answer_keyword_recall_avg", "kw_recall"),
        ("abstention_rate", "abstention"),
        ("rejection_accuracy", "rejection_acc"),
        ("avg_vlm_pages_required", "vlm_pages(누적X,질문당)"),
        ("avg_vlm_calls", "vlm_calls(참고)"),
        ("avg_selected_doc_count", "avg_docs"),
        ("avg_elapsed_seconds", "avg_sec"),
    ]

    for bucket in _BUCKET_SOURCES:
        rows = [compare_result["summary"][mode][bucket] for mode in compare_result["modes"]]
        if all(r["total_questions"] == 0 for r in rows):
            continue
        lines.append(f"## {bucket}")
        lines.append("")
        header = "| mode | " + " | ".join(label for _, label in cols) + " |"
        sep = "|---|" + "---|" * len(cols)
        lines.append(header)
        lines.append(sep)
        for mode, row in zip(compare_result["modes"], rows):
            cells = [_fmt(row.get(key)) for key, _ in cols]
            lines.append(f"| {mode} | " + " | ".join(cells) + " |")
        lines.append("")

    if compare_result.get("stratified"):
        stratify_field = compare_result["stratified_by"]
        lines.append(f"## 층화: {stratify_field}")
        lines.append("")
        all_keys = sorted({k for mode_groups in compare_result["stratified"].values() for k in mode_groups})
        for key in all_keys:
            lines.append(f"### {stratify_field} = {key}")
            lines.append("")
            header = "| mode | " + " | ".join(label for _, label in cols) + " |"
            sep = "|---|" + "---|" * len(cols)
            lines.append(header)
            lines.append(sep)
            for mode in compare_result["modes"]:
                row = compare_result["stratified"][mode].get(key)
                if row is None:
                    continue
                cells = [_fmt(row.get(k)) for k, _ in cols]
                lines.append(f"| {mode} | " + " | ".join(cells) + " |")
            lines.append("")

    return "\n".join(lines)


def save_compare(compare_result: dict[str, Any], config: Config) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.output_dir / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"compare_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(compare_result, f, ensure_ascii=False, indent=2, allow_nan=False)

    md_path = out_dir / f"compare_{ts}.md"
    md_path.write_text(render_compare_markdown(compare_result), encoding="utf-8")

    logger.info("비교 결과 저장: %s / %s", json_path, md_path)
    return json_path, md_path

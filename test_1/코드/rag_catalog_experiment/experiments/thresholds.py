"""모드별 거절 게이트(min_dense_similarity류)의 dense 유사도 분포를 실측해 리포트.

세 모드는 서로 다른 텍스트 분포(카탈로그 설명문/파일명/페이지 요약) 위에서 동작하므로
같은 임계값을 공유하면 비교가 무의미하다(retrieval.py 모듈 docstring 참조). 그래서
"같은 절대 임계값 하나"를 찾는 대신, 모드별로 자신의 분포에서 스윕해 **정답 문서가 있는
질문의 최소 재현율(retention)**을 기준으로 삼는 임계값과, 그 지점에서 무관 질문을 얼마나
걸러내는지(rejection)를 함께 보고한다. 자동으로 config를 덮어쓰지 않는다 — 리포트만 낸다.

`depth="docs"`로 돌기 때문에 VLM을 전혀 호출하지 않는다 (질의분석 LLM 1회 + 임베딩만).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import Config
from ..models import Backend

logger = logging.getLogger(__name__)

_GATE_ATTR = {
    "catalog": "min_dense_similarity",
    "filename_only": "min_filename_dense_similarity",
    "no_catalog": "min_page_dense_similarity",
}


def _dense_similarity_direct(question: str, config: Config, backend: Backend, mode: str) -> float | None:
    """selection_score(RRF)가 아니라 실제 dense cosine 유사도를 얻기 위해 인덱스를 직접 질의."""
    from ..indexes import get_index
    from ..retrieval import analyze_query

    qa = analyze_query(question, backend)
    query_text = qa.get("retrieval_query") or question

    index_name = {"catalog": "catalog_index", "filename_only": "filename_index", "no_catalog": "page_index"}[mode]
    index = get_index(index_name, config, backend)
    results = index.query(query_text, n_results=1)
    if not results:
        return None
    return results[0].dense_similarity


def build_threshold_report(
    eval_items: list[dict], config: Config, backend: Backend, *, modes: list[str]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in eval_items:
        question = item["question"]
        question_type = item.get("question_type", "")
        is_irrelevant = question_type == "irrelevant"
        has_expected_doc = bool(item.get("expected_documents"))
        for mode in modes:
            sim = _dense_similarity_direct(question, config, backend, mode)
            rows.append(
                {
                    "id": item["id"],
                    "mode": mode,
                    "dense_similarity": sim,
                    "is_irrelevant": is_irrelevant,
                    "has_expected_doc": has_expected_doc,
                    "current_gate": getattr(config, _GATE_ATTR[mode]),
                }
            )

    report: dict[str, Any] = {"gate_attr": _GATE_ATTR, "rows": rows, "by_mode": {}}
    for mode in modes:
        mode_rows = [r for r in rows if r["mode"] == mode and r["dense_similarity"] is not None]
        relevant_sims = sorted(r["dense_similarity"] for r in mode_rows if not r["is_irrelevant"])
        irrelevant_sims = sorted(r["dense_similarity"] for r in mode_rows if r["is_irrelevant"])

        # relevant 질문의 95%를 유지하는 임계값(하위 5% 컷) — 공통 작동점.
        retained_95_threshold = None
        if relevant_sims:
            cut_idx = max(0, int(len(relevant_sims) * 0.05) - 1) if len(relevant_sims) > 1 else 0
            retained_95_threshold = relevant_sims[cut_idx] if len(relevant_sims) > 1 else relevant_sims[0]

        rejected_at_95 = None
        if irrelevant_sims and retained_95_threshold is not None:
            rejected_at_95 = sum(1 for s in irrelevant_sims if s < retained_95_threshold) / len(irrelevant_sims)

        report["by_mode"][mode] = {
            "current_gate": getattr(config, _GATE_ATTR[mode]),
            "n_relevant": len(relevant_sims),
            "n_irrelevant": len(irrelevant_sims),
            "relevant_min": relevant_sims[0] if relevant_sims else None,
            "relevant_max": relevant_sims[-1] if relevant_sims else None,
            "irrelevant_min": irrelevant_sims[0] if irrelevant_sims else None,
            "irrelevant_max": irrelevant_sims[-1] if irrelevant_sims else None,
            "threshold_at_95pct_relevant_retention": retained_95_threshold,
            "irrelevant_rejection_rate_at_that_threshold": rejected_at_95,
        }
    return report


def render_threshold_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 게이트 임계값 분포 리포트",
        "",
        "모드마다 다른 텍스트 분포 위에서 게이트가 동작하므로, 같은 절대값을 공유하지 않는다. "
        "\"정상 질문 95% 유지\"라는 공통 작동점에서 무관 질문 거절률을 비교한다.",
        "",
        "| mode | 현재 게이트 | relevant min~max | irrelevant min~max | 95%유지 임계값(제안) | 그 임계값에서 무관질문 거절률 |",
        "|---|---|---|---|---|---|",
    ]
    for mode, m in report["by_mode"].items():
        rel = f"{m['relevant_min']:.4f}~{m['relevant_max']:.4f}" if m["relevant_min"] is not None else "-"
        irr = f"{m['irrelevant_min']:.4f}~{m['irrelevant_max']:.4f}" if m["irrelevant_min"] is not None else "-"
        thr = f"{m['threshold_at_95pct_relevant_retention']:.4f}" if m["threshold_at_95pct_relevant_retention"] is not None else "-"
        rej = f"{m['irrelevant_rejection_rate_at_that_threshold']:.2%}" if m["irrelevant_rejection_rate_at_that_threshold"] is not None else "-"
        lines.append(f"| {mode} | {m['current_gate']:.2f} | {rel} | {irr} | {thr} | {rej} |")
    lines.append("")
    lines.append("자동 튜닝은 하지 않는다. 이 표를 보고 config.yaml의 게이트 값을 사람이 판단해 조정한다.")
    return "\n".join(lines)


def save_threshold_report(report: dict[str, Any], config: Config) -> tuple[Path, Path]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.output_dir / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"thresholds_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)

    md_path = out_dir / f"thresholds_{ts}.md"
    md_path.write_text(render_threshold_markdown(report), encoding="utf-8")

    logger.info("임계값 리포트 저장: %s / %s", json_path, md_path)
    return json_path, md_path

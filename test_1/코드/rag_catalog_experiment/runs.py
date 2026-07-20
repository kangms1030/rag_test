"""outputs/ 산출물 정리: 분류 이동, 검증, RUNS.md 갱신.

산출물은 삭제하지 않는다 — mock/failed/old_schema/success로 분류해서 이동만 한다.
evidence/{run_id}/ 이미지는 answer JSON이 절대경로로 참조하므로 옮기지 않는다.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config
from .schema import SCHEMA_VERSION, classify_answer

logger = logging.getLogger(__name__)

RUN_CATEGORIES = ("success", "failed", "mock", "old_schema")


@dataclass
class MovePlan:
    src: Path
    dst: Path
    category: str
    reason: str


def plan_answer_moves(config: Config) -> list[MovePlan]:
    """outputs/ 루트에 흩어진 answer_*.json을 분류해 이동 계획을 만든다 (실행 X)."""
    plans: list[MovePlan] = []
    for path in sorted(config.output_dir.glob("answer_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("%s: 읽기 실패, old_schema로 분류 (%s)", path, e)
            category = "old_schema"
        else:
            category = classify_answer(obj)
        dst_dir = config.output_dir / "runs" / category
        plans.append(
            MovePlan(src=path, dst=dst_dir / path.name, category=category, reason=f"classify_answer -> {category}")
        )
    return plans


def plan_evaluation_moves(config: Config) -> list[MovePlan]:
    """outputs/ 루트의 evaluation_*.json을 outputs/evaluations/로 이동 계획."""
    dst_dir = config.output_dir / "evaluations"
    return [
        MovePlan(src=path, dst=dst_dir / path.name, category="evaluations", reason="evaluation 결과 -> evaluations/")
        for path in sorted(config.output_dir.glob("evaluation_*.json"))
    ]


def clean_runs(config: Config, *, dry_run: bool = True) -> list[MovePlan]:
    plans = plan_answer_moves(config) + plan_evaluation_moves(config)
    for p in plans:
        if dry_run:
            logger.info("[dry-run] %s -> %s (%s)", p.src, p.dst, p.reason)
            continue
        p.dst.parent.mkdir(parents=True, exist_ok=True)
        if p.dst.exists():
            logger.warning("%s: 대상이 이미 존재해 건너뜀 (수동 확인 필요)", p.dst)
            continue
        shutil.move(str(p.src), str(p.dst))
        logger.info("이동: %s -> %s", p.src, p.dst)
    return plans


def _refresh_latest(config: Config) -> Path | None:
    """가장 최근 success run을 outputs/latest/에 복사 (심볼릭 링크 대신 복사본)."""
    success_dir = config.output_dir / "runs" / "success"
    if not success_dir.exists():
        return None
    candidates = sorted(success_dir.glob("answer_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    latest_dir = config.output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for old in latest_dir.glob("answer_*.json"):
        old.unlink()
    dst = latest_dir / candidates[0].name
    shutil.copy2(candidates[0], dst)
    return dst


def validate_outputs(config: Config) -> dict[str, Any]:
    """outputs/runs/success/* 전부가 최신 schema_version을 만족하는지 검사하고 latest/를 갱신."""
    success_dir = config.output_dir / "runs" / "success"
    results: list[dict[str, Any]] = []
    if success_dir.exists():
        for path in sorted(success_dir.glob("answer_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            ok = obj.get("schema_version") == SCHEMA_VERSION
            results.append({"file": str(path), "schema_version": obj.get("schema_version"), "valid": ok})

    run_counts = {}
    for cat in RUN_CATEGORIES:
        d = config.output_dir / "runs" / cat
        run_counts[cat] = len(list(d.glob("answer_*.json"))) if d.exists() else 0

    latest = _refresh_latest(config)
    return {
        "expected_schema_version": SCHEMA_VERSION,
        "success_runs_checked": len(results),
        "success_runs_valid": sum(1 for r in results if r["valid"]),
        "invalid": [r for r in results if not r["valid"]],
        "run_counts": run_counts,
        "latest_copied_to": str(latest) if latest else None,
    }


def write_runs_md(config: Config) -> Path:
    lines = [
        "# outputs/ 실행 이력",
        "",
        f"자동 생성됨 ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}). "
        "`clean-runs`/`validate-outputs` 실행 시 갱신된다.",
        "",
        "## runs/success/",
        "",
        "| run_id | retrieval_mode | question | 문서수 | 근거수 |",
        "|---|---|---|---|---|",
    ]
    success_dir = config.output_dir / "runs" / "success"
    if success_dir.exists():
        for path in sorted(success_dir.glob("answer_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            run_id = obj.get("run_id", path.stem.replace("answer_", ""))
            mode = obj.get("retrieval_mode", "?")
            q = str(obj.get("question", ""))[:40]
            docs = len(obj.get("selected_documents", []))
            ev = len(obj.get("page_evidence", []))
            lines.append(f"| {run_id} | {mode} | {q} | {docs} | {ev} |")
    lines.append("")

    for cat, note in (
        ("failed", "실패 실행"),
        ("mock", "모델 미호출 배선 검증"),
        ("old_schema", "schema_version 도입 이전 산출물 (REPORT.md 참조)"),
    ):
        d = config.output_dir / "runs" / cat
        n = len(list(d.glob("answer_*.json"))) if d.exists() else 0
        lines.append(f"## runs/{cat}/ — {note} ({n}개)")
        lines.append("")

    lines.append("## comparisons/ · evaluations/")
    lines.append("")
    comp_dir = config.output_dir / "comparisons"
    for c in sorted(comp_dir.glob("compare_*.md")) if comp_dir.exists() else []:
        lines.append(f"- `{c.relative_to(config.output_dir)}`")
    eval_dir = config.output_dir / "evaluations"
    for e in sorted(eval_dir.glob("evaluation_*.json")) if eval_dir.exists() else []:
        lines.append(f"- `{e.relative_to(config.output_dir)}`")
    lines.append("")

    out_path = config.output_dir / "RUNS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("RUNS.md 갱신: %s", out_path)
    return out_path

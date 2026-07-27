# -*- coding: utf-8 -*-
"""골든셋으로 v2 그래프 전체를 평가한다 (작업 7).

기존 ragcore/rag3/evaluate.py 와의 차이:
- 구버전 generate_answer 가 아니라 **메인그래프 ctx.graph.invoke** 를 탄다
  → 시나리오/FAQ 매칭 · clarify · composer · grader 까지 전부 평가 대상에 들어간다.
- 키워드 포함 여부만 보지 않고 **근거 정밀도**(컨텍스트 중 정답 근거 비중)를 측정한다.
  이 지표가 이번 개선의 핵심이다 — 트레이스 8e0815cd 에서 9.6% 였다.

실행:
    <intern_chatbot python> -X utf8 chatbot_demo_v2/scripts/run_eval.py
    ... --limit 5 --category ambiguous --tag baseline --no-cache

결과는 runtime/reports/eval_<tag>_<timestamp>.json 에 저장된다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT.parent))  # '챗봇' 폴더를 sys.path 에 → chatbot_demo_v2 패키지 임포트

from chatbot_demo_v2.config.settings import load_settings  # noqa: E402
from chatbot_demo_v2.app.dependencies import build_context  # noqa: E402

_PAGE_HDR = re.compile(r"--- (.+?) p(\d+) ---")


# ---------------------------------------------------------------- 지표 계산
def context_breakdown(answer_context: str) -> list[dict]:
    """answer_context 를 '--- {문서} p{n} ---' 헤더 기준으로 쪼개 페이지별 글자수를 낸다."""
    if not answer_context:
        return []
    out: list[dict] = []
    marks = list(_PAGE_HDR.finditer(answer_context))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(answer_context)
        body = answer_context[m.end():end]
        out.append({"document_name": m.group(1), "page_number": int(m.group(2)), "chars": len(body)})
    return out


def evidence_precision(parts: list[dict], expected_doc: str, expected_pages: list[int]) -> float | None:
    """컨텍스트 전체 글자수 중 정답 근거(문서·페이지)가 차지하는 비율.

    expected_doc 이 비어 있으면 None(측정 대상 아님). expected_pages 가 비어 있으면 문서만 본다.
    """
    if not expected_doc or not parts:
        return None
    total = sum(p["chars"] for p in parts) or 1
    hit = 0
    for p in parts:
        if expected_doc not in p["document_name"]:
            continue
        if expected_pages and p["page_number"] not in expected_pages:
            continue
        hit += p["chars"]
    return round(hit / total, 4)


def max_doc_share(parts: list[dict]) -> float | None:
    """한 문서가 컨텍스트에서 차지하는 최대 비율(문서 다양성 지표). 낮을수록 다양."""
    if not parts:
        return None
    total = sum(p["chars"] for p in parts) or 1
    by_doc: dict[str, int] = {}
    for p in parts:
        by_doc[p["document_name"]] = by_doc.get(p["document_name"], 0) + p["chars"]
    return round(max(by_doc.values()) / total, 4)


def evidence_recall(evidence: list[dict], expected_doc: str, expected_pages: list[int]) -> bool | None:
    """정답 문서(·페이지)가 근거 목록에 들어왔는가."""
    if not expected_doc:
        return None
    for ev in evidence or []:
        if expected_doc not in (ev.get("document_name") or ""):
            continue
        if expected_pages and ev.get("page_number") not in expected_pages:
            continue
        return True
    return False


# ---------------------------------------------------------------- 1문항 실행
def run_item(ctx, item: dict, *, judge: bool) -> dict:
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401  (문서화용)

    sid = f"eval-{item['id']}"
    thread = ctx.session_registry.thread_id(sid)
    init = {
        "session_id": sid,
        "thread_id": thread,
        "input_type": "text",
        "user_input": item["question"],
    }
    cfg = {"configurable": {"thread_id": thread}}

    t0 = time.time()
    error = None
    try:
        result = ctx.graph.invoke(init, cfg)
    except Exception as exc:  # noqa: BLE001
        return {
            "id": item["id"], "category": item["category"], "question": item["question"],
            "error": f"{type(exc).__name__}: {exc}", "elapsed_s": round(time.time() - t0, 2),
            "route": None, "route_ok": False,
        }
    elapsed = round(time.time() - t0, 2)

    # clarify 는 interrupt 로 멈춘다 → route 를 'clarify' 로 본다
    interrupted = bool(result.get("__interrupt__"))
    route = "clarify" if interrupted else (result.get("route") or None)

    rag = result.get("rag_result") or {}
    parts = context_breakdown(rag.get("answer_context") or "")
    exp_doc = item.get("expected_doc") or ""
    exp_pages = item.get("expected_pages") or []
    answer = (result.get("final_answer") or "") if not interrupted else ""

    metrics = (rag.get("metrics") or {})
    row = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected_route": item["expected_route"],
        "route": route,
        "route_ok": route == item["expected_route"],
        "elapsed_s": elapsed,
        "confidence": result.get("confidence"),
        "answer_path": result.get("answer_path"),
        "answer_len": len(answer),
        "answer_head": answer[:180],
        "composed": bool(result.get("composed")),
        "grader_verdict": result.get("grader_verdict"),
        "warnings": result.get("warnings") or [],
        # 근거 지표
        "evidence_precision": evidence_precision(parts, exp_doc, exp_pages),
        "evidence_recall": evidence_recall(result.get("evidence") or [], exp_doc, exp_pages),
        "max_doc_share": max_doc_share(parts),
        "context_chars": sum(p["chars"] for p in parts) if parts else 0,
        "context_pages": len(parts),
        "context_breakdown": parts,
        "evidence": [
            {"document_name": e.get("document_name"), "page_number": e.get("page_number")}
            for e in (result.get("evidence") or [])
        ],
        # 비용/호출
        "model_calls": metrics.get("total_model_calls"),
        "gemini_calls": metrics.get("gemini_calls"),
        "rerank_top_score": rag.get("rerank_top_score"),
        "rollback_history": rag.get("rollback_history") or [],
        "scenario_match": result.get("scenario_match"),
        "trace": [t.get("node") for t in (result.get("trace") or [])],
        "error": error,
    }

    # must_include
    musts = item.get("must_include") or []
    row["must_include_missing"] = [m for m in musts if m.lower() not in answer.lower()] if answer else musts
    row["must_include_ok"] = not row["must_include_missing"]

    # FAQ id 일치
    if item.get("expected_faq_id"):
        sm = result.get("scenario_match") or {}
        row["faq_id"] = sm.get("matched_id")
        row["faq_id_ok"] = sm.get("matched_id") == item["expected_faq_id"] and route == "faq"

    # LLM 해결도 판정(선택) — answer_grader 프롬프트 재사용
    if judge and answer and ctx.llm is not None and ctx.prompts is not None:
        raw = (ctx.llm.chat(ctx.prompts.render(
            "answer_grader", question=item["question"], answer=answer)) or "").upper()
        row["judge"] = "unresolved" if "UNRESOLVED" in raw else ("resolved" if "RESOLVED" in raw else None)

    return row


# ---------------------------------------------------------------- 집계
def summarize(rows: list[dict]) -> dict:
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    by_cat: dict[str, dict] = {}
    for r in rows:
        c = by_cat.setdefault(r["category"], {"n": 0, "route_ok": 0, "items": []})
        c["n"] += 1
        c["route_ok"] += 1 if r.get("route_ok") else 0
        c["items"].append(r["id"])
    for c in by_cat.values():
        c["route_acc"] = round(c["route_ok"] / c["n"], 3)

    return {
        "n": len(rows),
        "route_accuracy": round(sum(1 for r in rows if r.get("route_ok")) / max(len(rows), 1), 4),
        "evidence_precision_avg": avg([r.get("evidence_precision") for r in rows]),
        "evidence_recall_rate": avg([1.0 if r.get("evidence_recall") else 0.0
                                     for r in rows if r.get("evidence_recall") is not None]),
        "max_doc_share_avg": avg([r.get("max_doc_share") for r in rows]),
        "must_include_pass": round(sum(1 for r in rows if r.get("must_include_ok")) / max(len(rows), 1), 4),
        "elapsed_avg_s": avg([r.get("elapsed_s") for r in rows]),
        "elapsed_total_s": round(sum(r.get("elapsed_s") or 0 for r in rows), 1),
        "gemini_calls_total": sum(r.get("gemini_calls") or 0 for r in rows),
        "rollback_count": sum(len(r.get("rollback_history") or []) for r in rows),
        "errors": [r["id"] for r in rows if r.get("error")],
        "by_category": by_cat,
    }


def print_report(summary: dict, rows: list[dict]) -> None:
    s = summary
    print()
    print("=" * 96)
    print("문항 %d개 | 라우트 정확도 %.1f%% | 근거정밀도 %s | 근거회수 %s | 최대문서독점 %s"
          % (s["n"], 100 * s["route_accuracy"],
             "-" if s["evidence_precision_avg"] is None else "%.1f%%" % (100 * s["evidence_precision_avg"]),
             "-" if s["evidence_recall_rate"] is None else "%.1f%%" % (100 * s["evidence_recall_rate"]),
             "-" if s["max_doc_share_avg"] is None else "%.1f%%" % (100 * s["max_doc_share_avg"])))
    print("must_include 통과 %.1f%% | 평균 %.1fs | 총 %.0fs | Gemini %d콜 | 롤백 %d회"
          % (100 * s["must_include_pass"], s["elapsed_avg_s"] or 0, s["elapsed_total_s"],
             s["gemini_calls_total"], s["rollback_count"]))
    print("=" * 96)
    print("%-14s %5s %8s" % ("카테고리", "문항", "라우트정확도"))
    for cat, c in sorted(s["by_category"].items()):
        print("  %-12s %5d %7.1f%%" % (cat, c["n"], 100 * c["route_acc"]))
    print()
    print("%-14s %-9s %-9s %6s %6s %6s  %s" % ("id", "기대", "실제", "근거P", "독점", "초", "비고"))
    for r in rows:
        ep = "-" if r.get("evidence_precision") is None else "%.0f%%" % (100 * r["evidence_precision"])
        ms = "-" if r.get("max_doc_share") is None else "%.0f%%" % (100 * r["max_doc_share"])
        note = []
        if r.get("error"):
            note.append("ERR " + r["error"][:40])
        if r.get("must_include_missing"):
            note.append("누락:" + ",".join(r["must_include_missing"])[:24])
        if r.get("rollback_history"):
            note.append("롤백%d" % len(r["rollback_history"]))
        if r.get("faq_id_ok") is False:
            note.append("faq!=%s" % (r.get("faq_id") or "None"))
        print("%-14s %-9s %-9s %6s %6s %6.1f  %s"
              % (r["id"], r.get("expected_route"), r.get("route") or "-", ep, ms,
                 r.get("elapsed_s") or 0, " ".join(note)))
    print()


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="v2 그래프 골든셋 평가")
    ap.add_argument("--eval-set", default=str(PKG_ROOT / "data" / "eval_set.json"))
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N문항만")
    ap.add_argument("--category", default="", help="카테고리 필터(쉼표 구분)")
    ap.add_argument("--tag", default="run", help="결과 파일 태그")
    ap.add_argument("--judge", action="store_true", help="LLM 해결도 판정 추가(호출 1회/문항)")
    ap.add_argument("--no-cache", action="store_true", help="RAG TTL 캐시 끄기(정확한 지연 측정)")
    ap.add_argument("--out-dir", default=str(PKG_ROOT / "runtime" / "reports"))
    args = ap.parse_args()

    data = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    items = data["items"]
    if args.category:
        cats = {c.strip() for c in args.category.split(",")}
        items = [i for i in items if i["category"] in cats]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("대상 문항 없음")
        return 1

    settings = load_settings()
    if args.no_cache:
        object.__setattr__(settings, "rag_cache_ttl_s", 0)  # frozen dataclass 우회
    print("설정: backend=%s cache_ttl=%ss clarify=%s grader=%s"
          % (settings.rag_backend, settings.rag_cache_ttl_s, settings.clarify_enabled, settings.grader_enabled))
    print("문항 %d개 실행..." % len(items))

    ctx = build_context(settings)
    try:
        ctx.rag_adapter.warm_up(deep=False)
    except Exception as exc:  # noqa: BLE001
        print("[경고] 엔진 워밍업 실패:", type(exc).__name__, exc)

    rows = []
    for n, item in enumerate(items, 1):
        print("  [%2d/%d] %-14s %s" % (n, len(items), item["id"], item["question"][:52]), flush=True)
        rows.append(run_item(ctx, item, judge=args.judge))

    summary = summarize(rows)
    print_report(summary, rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("eval_%s_%s.json" % (args.tag, datetime.now().strftime("%Y%m%dT%H%M%S")))
    out.write_text(json.dumps({
        "tag": args.tag,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "rag_backend": settings.rag_backend,
            "rag_cache_ttl_s": settings.rag_cache_ttl_s,
            "clarify_enabled": settings.clarify_enabled,
            "clarify_min_score": settings.clarify_min_score,
            "scenario_match_threshold": settings.scenario_match_threshold,
            "grader_enabled": settings.grader_enabled,
        },
        "summary": summary,
        "rows": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

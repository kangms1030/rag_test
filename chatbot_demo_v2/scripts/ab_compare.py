# -*- coding: utf-8 -*-
"""평가 리포트 2개(이상)를 나란히 비교한다 (작업 7).

run_eval.py 가 남긴 runtime/reports/eval_*.json 을 읽어 지표 차이와 **문항별 변화**를 낸다.
착수 전 조사에서 "유력해 보이는 개선 4개 중 3개가 실측상 악화"였으므로, 어떤 설정 변경도
이 스크립트로 전/후를 비교해 판정한 뒤에만 채택한다.

실행:
    <python> -X utf8 chatbot_demo_v2/scripts/ab_compare.py baseline after-work1
    ... --reports-dir <경로>          # 기본 runtime/reports
    ... A.json B.json                 # 파일 경로를 직접 줘도 된다
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PKG_ROOT / "runtime" / "reports"

# (키, 표시명, 백분율여부, 클수록좋은가)
METRICS = [
    ("route_accuracy", "라우트 정확도", True, True),
    ("evidence_precision_avg", "근거 정밀도", True, True),
    ("evidence_recall_rate", "근거 회수율", True, True),
    ("max_doc_share_avg", "최대 문서독점", True, False),
    ("must_include_pass", "must_include", True, True),
    ("elapsed_avg_s", "평균 지연(s)", False, False),
    ("elapsed_total_s", "총 지연(s)", False, False),
    ("gemini_calls_total", "Gemini 콜", False, False),
    ("rollback_count", "롤백 횟수", False, False),
]


def resolve(token: str, reports_dir: Path) -> Path:
    """'baseline' 같은 태그면 해당 태그의 최신 리포트를, 경로면 그대로."""
    p = Path(token)
    if p.is_file():
        return p
    cands = sorted(reports_dir.glob(f"eval_{token}_*.json"))
    if not cands:
        raise SystemExit(f"리포트를 찾을 수 없음: {token} (dir={reports_dir})")
    return cands[-1]


def fmt(val, pct: bool) -> str:
    if val is None:
        return "-"
    return ("%.1f%%" % (100 * val)) if pct else ("%.1f" % val if isinstance(val, float) else str(val))


def delta_str(a, b, pct: bool, higher_better: bool) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    if abs(d) < 1e-9:
        return "  ="
    mark = "↑" if d > 0 else "↓"
    good = (d > 0) == higher_better
    tag = "개선" if good else "악화"
    shown = ("%+.1f%%p" % (100 * d)) if pct else ("%+.1f" % d)
    return "  %s %s %s" % (mark, shown, tag)


def main() -> int:
    ap = argparse.ArgumentParser(description="평가 리포트 A/B 비교")
    ap.add_argument("reports", nargs="+", help="태그 또는 파일경로 (2개 이상)")
    ap.add_argument("--reports-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--show-items", action="store_true", help="문항별 변화 전부 표시")
    args = ap.parse_args()

    rdir = Path(args.reports_dir)
    paths = [resolve(t, rdir) for t in args.reports]
    data = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    names = [d.get("tag") or p.stem for d, p in zip(data, paths)]

    print()
    for n, p, d in zip(names, paths, data):
        print("[%s] %s  (문항 %d, %s)" % (n, p.name, d["summary"]["n"], d.get("generated_at", "")))
    print()

    # --- 지표 표 ---
    w = 16
    print("%-16s" % "지표" + "".join("%*s" % (w, n[:w - 1]) for n in names) + "   변화(첫→마지막)")
    print("-" * (16 + w * len(names) + 22))
    for key, label, pct, hb in METRICS:
        vals = [d["summary"].get(key) for d in data]
        row = "%-16s" % label + "".join("%*s" % (w, fmt(v, pct)) for v in vals)
        print(row + delta_str(vals[0], vals[-1], pct, hb))
    print()

    # --- 카테고리별 라우트 정확도 ---
    cats = sorted({c for d in data for c in d["summary"]["by_category"]})
    print("%-16s" % "카테고리 라우트" + "".join("%*s" % (w, n[:w - 1]) for n in names) + "   변화")
    print("-" * (16 + w * len(names) + 22))
    for c in cats:
        vals = [d["summary"]["by_category"].get(c, {}).get("route_acc") for d in data]
        row = "%-16s" % c + "".join("%*s" % (w, fmt(v, True)) for v in vals)
        print(row + delta_str(vals[0], vals[-1], True, True))
    print()

    # --- 문항별 변화 (첫 리포트 vs 마지막) ---
    first = {r["id"]: r for r in data[0]["rows"]}
    last = {r["id"]: r for r in data[-1]["rows"]}
    common = [i for i in first if i in last]

    changed_route = [(i, first[i].get("route"), last[i].get("route")) for i in common
                     if first[i].get("route") != last[i].get("route")]
    fixed = [i for i in common if not first[i].get("route_ok") and last[i].get("route_ok")]
    broke = [i for i in common if first[i].get("route_ok") and not last[i].get("route_ok")]

    print("라우트 개선 %d문항: %s" % (len(fixed), ", ".join(fixed) or "-"))
    print("라우트 악화 %d문항: %s" % (len(broke), ", ".join(broke) or "-"))
    if broke:
        print("  ⚠ 악화 문항이 있으면 채택 전에 반드시 원인을 확인할 것")
    print()

    if changed_route:
        print("라우트가 바뀐 문항:")
        for i, a, b in changed_route:
            print("  %-14s %-9s → %-9s (기대 %s)" % (i, a or "-", b or "-", first[i].get("expected_route")))
        print()

    # 근거 정밀도 변화
    ep_rows = [(i, first[i].get("evidence_precision"), last[i].get("evidence_precision"))
               for i in common
               if first[i].get("evidence_precision") is not None
               and last[i].get("evidence_precision") is not None]
    ep_rows = [r for r in ep_rows if abs((r[2] or 0) - (r[1] or 0)) > 0.01]
    if ep_rows:
        ep_rows.sort(key=lambda r: (r[2] or 0) - (r[1] or 0))
        print("근거 정밀도가 바뀐 문항 (악화 순):")
        for i, a, b in ep_rows:
            print("  %-14s %5.1f%% → %5.1f%%   %+.1f%%p" % (i, 100 * a, 100 * b, 100 * (b - a)))
        print()

    if args.show_items:
        print("%-14s %-9s %-9s %8s %8s" % ("id", names[0][:8], names[-1][:8], "근거P", "지연"))
        for i in common:
            a, b = first[i], last[i]
            print("%-14s %-9s %-9s %5s→%-5s %5.1f→%-5.1f"
                  % (i, a.get("route") or "-", b.get("route") or "-",
                     fmt(a.get("evidence_precision"), True), fmt(b.get("evidence_precision"), True),
                     a.get("elapsed_s") or 0, b.get("elapsed_s") or 0))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

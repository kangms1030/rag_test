"""rag3x 평가 하버스트 — controller_x를 x_overrides로 돌려 baseline(varfix)과 비교.

용법:
  python tmp/x_eval.py --tag B_local_all --x fail_fast=1,verify_skip=1,trim=1
  python tmp/x_eval.py --tag tail --qids core_002,core_008,vp_006,vp_007,vp_013,core_001,core_004,vp_009
  python tmp/x_eval.py --tag B_full --x fail_fast=1,verify_skip=1,trim=1 --all

x 단축키: fail_fast->x_fail_fast_on_length_budget, verify_skip->x_conditional_verify_skip,
          trim->x_adaptive_trim, backend=gemini->x_backend, gvision=1->x_gemini_vision.
결과: probes/results/PHASE6_<tag>.json (resume 지원). baseline과 문항단위 diff + 절대지표 게이트.
"""
from __future__ import annotations
import argparse, io, json, sys, time
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

TEST3 = Path(__file__).resolve().parents[1]
ROOT = TEST3.parent
sys.path.insert(0, str(TEST3))

from rag3x.xconfig import load_x_config
from rag3x.backends import get_x_backend
from rag3x.controller_x import answer_question

DATASETS = [ROOT / "rag_test" / "test_2" / "rag_eval_dataset.json",
            ROOT / "test12_total_test" / "test2_vlm_probe" / "vlm_probe_dataset.json"]
IRRELEVANT_IDS = {"irrelevant_001", "irrelevant_002", "irrelevant_003", "vi_001", "vi_002"}
BASE = TEST3 / "probes" / "results" / "varfix_eval.json"

_XKEYS = {
    "fail_fast": "x_fail_fast_on_length_budget",
    "verify_skip": "x_conditional_verify_skip",
    "trim": "x_adaptive_trim",
    "backend": "x_backend",
    "gvision": "x_gemini_vision",
    "verify_skip_tau": "x_verify_skip_tau",
    "trim_ratio": "x_adaptive_trim_drop_ratio",
    "decompose": "x_enable_decompose_routing",
    "citation": "x_sentence_citation_verify",
}


def parse_x(s: str) -> dict:
    out = {}
    if not s:
        return out
    for kv in s.split(","):
        k, _, v = kv.partition("=")
        k = k.strip()
        key = _XKEYS.get(k, k)
        v = v.strip()
        if v in ("1", "true", "True"):
            out[key] = True
        elif v in ("0", "false", "False"):
            out[key] = False
        else:
            try:
                out[key] = float(v) if "." in v else int(v)
            except ValueError:
                out[key] = v
    return out


def load_items(datasets=None):
    items = {}
    for p in (datasets or DATASETS):
        ds = json.loads(Path(p).read_text(encoding="utf-8"))
        arr = ds if isinstance(ds, list) else list(ds.values())[0]
        for q in arr:
            items[q["id"]] = q
    return items


def doc_match(dn, exp):
    return any(d == dn or Path(d).name == dn or d in dn or dn in d for d in exp)


def norm(s):
    return "".join(str(s).split()).lower()


def kw_hit(ans, kws):
    if not kws:
        return None
    a = norm(ans)
    return sum(1 for k in kws if norm(k) in a) / len(kws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--x", default="")
    ap.add_argument("--qids", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="기존 결과 무시하고 새로 시작")
    ap.add_argument("--no-warmup", action="store_true", help="워밍업 생략(콜드 편차 측정용)")
    ap.add_argument("--dataset", default="", help="커스텀 평가셋 JSON(종합형 등). 지정 시 그 안 전 문항 실행")
    args = ap.parse_args()

    x_over = parse_x(args.x)
    xcfg = load_x_config(x_overrides=x_over)
    backend = get_x_backend(xcfg)
    items = load_items([args.dataset] if args.dataset else None)

    # 워밍업 (§6.4: warm 비교 필수) — 리랭커/임베딩/백엔드를 예열해 첫 문항 콜드로드 제거.
    if not args.no_warmup:
        from rag3.rerank import get_reranker
        from rag3.flat_index import get_flat_chunk_index
        wt = time.time()
        get_reranker(xcfg)
        get_flat_chunk_index(xcfg, backend).count()
        backend.embed(["워밍업"], is_query=True)
        backend.chat_text("답변은 '준비'라고만 하세요.")
        print(f"[warmup] {time.time()-wt:.1f}s (리랭커+임베딩+백엔드 예열)")

    if args.dataset:
        order = list(items.keys())  # 커스텀셋 전 문항
    elif args.all or not args.qids:
        base_all = json.loads(BASE.read_text(encoding="utf-8"))["results"]
        order = [r["qid"] for r in base_all]  # baseline과 같은 36문항·순서
    else:
        order = [q.strip() for q in args.qids.split(",") if q.strip()]

    base = {r["qid"]: r for r in json.loads(BASE.read_text(encoding="utf-8"))["results"]}
    OUT = TEST3 / "probes" / "results" / f"PHASE6_{args.tag}.json"

    results, done = [], set()
    if OUT.exists() and not args.fresh:
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
            results = prev.get("results", [])
            done = {r["qid"] for r in results}
            print(f"resume: 기존 {len(done)}문항 건너뜀")
        except Exception:
            results, done = [], set()

    print(f"x_overrides={x_over}  backend={getattr(xcfg,'x_backend','ollama')}  n={len(order)}")
    for i, qid in enumerate(order):
        if qid in done or qid not in items:
            continue
        q = items[qid]
        exp_docs = q.get("expected_documents", [])
        exp_pages = q.get("expected_pages", [])
        kws = q.get("expected_answer_keywords", [])
        is_irrel = qid in IRRELEVANT_IDS or not exp_pages
        t0 = time.time()
        r = answer_question(q["question"], xcfg, backend, run_id=qid)
        dt = time.time() - t0
        pages = r["selected_pages"]
        ev = any(doc_match(p["document_name"], exp_docs) and p["page_number"] in exp_pages for p in pages[:3])
        abstain = (r["answer_path"] == "none") or (r["confidence"] == "abstain")
        verif = r.get("verification") or {}
        rec = {"qid": qid, "irrelevant": is_irrel, "answer_path": r["answer_path"],
               "confidence": r["confidence"], "evidence_present": ev, "abstain": abstain,
               "kw_hit": kw_hit(r["final_answer"], kws),
               "unsupported_claims": verif.get("unsupported_claims", []),
               "groundedness": verif.get("groundedness"),
               "model_calls": r["metrics"]["total_model_calls"],
               "length_retry": r["metrics"].get("length_retry_count", 0),
               "rollback": [h.get("action") for h in r.get("rollback_history", [])],
               "elapsed_s": round(dt, 1), "final_answer_head": (r["final_answer"] or "")[:100],
               "cost": r["metrics"].get("gemini_cost"),
               "gemini_api_s": r["metrics"].get("gemini_api_s"),
               "gemini_calls": r["metrics"].get("gemini_calls")}
        results.append(rec)
        OUT.write_text(json.dumps({"partial": True, "x_overrides": x_over, "results": results},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        b = base.get(qid, {})
        bk = b.get("kw_hit")
        be = b.get("elapsed_s")
        tag = "무관" if is_irrel else f"ev={int(ev)}"
        print(f"[{i+1}/{len(order)}] {qid:14s} {tag} path={r['answer_path']:6s} conf={r['confidence']:8s} "
              f"kw={rec['kw_hit']}(기준{bk}) lr={rec['length_retry']} calls={rec['model_calls']} "
              f"rb={rec['rollback']} {dt:.0f}s(기준{be})")

    # 요약 + 절대지표 게이트
    rel = [r for r in results if not r["irrelevant"]]
    irr = [r for r in results if r["irrelevant"]]
    ev_present = [r for r in rel if r["evidence_present"]]
    answered = [r for r in rel if not r["abstain"]]
    kw_vals = [r["kw_hit"] for r in rel if r["kw_hit"] is not None]
    # 환각 프록시: 답변했는데(비-abstain) 미지원 숫자/코드 주장이 남은 문항
    hallucination_suspects = [r["qid"] for r in answered if r.get("unsupported_claims")]
    summary = {
        "n": len(results), "n_relevant": len(rel), "n_irrelevant": len(irr),
        "evidence_present": len(ev_present),
        "answered_rate": round(len(answered) / len(rel), 4) if rel else None,
        "avg_kw_hit": round(sum(kw_vals) / len(kw_vals), 4) if kw_vals else None,
        "irrelevant_reject_rate": round(sum(1 for r in irr if r["abstain"]) / len(irr), 4) if irr else None,
        "hallucination_suspects": hallucination_suspects,
        "avg_model_calls": round(sum(r["model_calls"] for r in rel) / len(rel), 2) if rel else None,
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / len(results), 1) if results else None,
        "max_elapsed_s": round(max(r["elapsed_s"] for r in results), 1) if results else None,
        "avg_gemini_api_s": (round(sum(r["gemini_api_s"] for r in results if r.get("gemini_api_s")) /
                                   sum(1 for r in results if r.get("gemini_api_s")), 2)
                             if any(r.get("gemini_api_s") for r in results) else None),
        "avg_gemini_calls": (round(sum(r["gemini_calls"] for r in results if r.get("gemini_calls")) /
                                   sum(1 for r in results if r.get("gemini_calls")), 2)
                             if any(r.get("gemini_calls") for r in results) else None),
        "total_cost": round(sum(r["cost"] for r in results if r.get("cost")), 6) or None,
    }
    # baseline 대비 문항 diff
    improved, regressed, faster = [], [], []
    for r in results:
        b = base.get(r["qid"])
        if not b:
            continue
        nk, bk = (r["kw_hit"] or 0), (b.get("kw_hit") or 0)
        if nk > bk + 1e-9:
            improved.append((r["qid"], bk, nk))
        elif nk < bk - 1e-9:
            regressed.append((r["qid"], bk, nk))
        if b.get("elapsed_s") and r["elapsed_s"] < b["elapsed_s"] - 1e-9:
            faster.append((r["qid"], b["elapsed_s"], r["elapsed_s"]))
    summary["kw_improved_vs_base"] = improved
    summary["kw_regressed_vs_base"] = regressed
    summary["faster_vs_base"] = faster
    OUT.write_text(json.dumps({"tag": args.tag, "x_overrides": x_over, "summary": summary,
                               "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== PHASE6_{args.tag} 요약 ===")
    print(f"  n={summary['n']} avg_kw={summary['avg_kw_hit']} 답변율={summary['answered_rate']} "
          f"무관거절={summary['irrelevant_reject_rate']}")
    print(f"  환각의심(미지원주장): {summary['hallucination_suspects']}")
    print(f"  지연 avg {summary['avg_elapsed_s']}s / max {summary['max_elapsed_s']}s  호출 {summary['avg_model_calls']}")
    print(f"  kw개선 {improved}  kw회귀 {regressed}")
    print(f"  비용 {summary['total_cost']}")
    print(f"저장: {OUT}")
    if regressed:
        print("  ⚠ kw 회귀 문항 존재 — 채택 전 검토 필요")


if __name__ == "__main__":
    main()

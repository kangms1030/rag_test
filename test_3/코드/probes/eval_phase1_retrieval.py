"""Phase 1 게이트 검증: 실제 retrieve v2 파이프라인을 36문항에 돌려 검색 품질 측정.

측정: page_hit@1/@3, doc_hit@1/@3, 무관 거절률, rerank_top_score 분포(floor 캘리브레이션용).
통과 기준(계획): page_hit@3 >= 0.75, doc_hit >= 0.92, 무관거절 5/5 유지.
결과: test_3/probes/results/phase1_retrieval.json
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test_3"))
from rag3.config import load_config       # noqa: E402
from rag3.models import get_backend       # noqa: E402
from rag3.retrieve import run_retrieval    # noqa: E402

DATASETS = [
    ROOT / "rag_test" / "test_2" / "rag_eval_dataset.json",
    ROOT / "test12_total_test" / "test2_vlm_probe" / "vlm_probe_dataset.json",
]
OUT = Path(__file__).resolve().parent / "results" / "phase1_retrieval.json"

IRRELEVANT_IDS = {"irrelevant_001", "irrelevant_002", "irrelevant_003", "vi_001", "vi_002"}


def load_items():
    items = []
    for p in DATASETS:
        ds = json.loads(p.read_text(encoding="utf-8"))
        arr = ds if isinstance(ds, list) else list(ds.values())[0]
        items.extend(arr)
    return items


def doc_match(dn: str, exp_docs) -> bool:
    return any(d == dn or Path(d).name == dn or d in dn or dn in d for d in exp_docs)


def main():
    cfg = load_config()
    backend = get_backend(cfg)
    print(f"use_catalog_gate={cfg.use_catalog_gate} rerank_floor={cfg.rerank_score_floor} "
          f"candidates={cfg.retrieve_candidates} final_pages={cfg.final_pages}")

    items = load_items()
    per_q = []
    agg = {"page_hit@1": [], "page_hit@3": [], "doc_hit@1": [], "doc_hit@3": []}
    irrel_reject = []
    rel_top_scores = []
    irrel_top_scores = []

    for q in items:
        qid = q["id"]
        exp_docs = q.get("expected_documents", [])
        exp_pages = q.get("expected_pages", [])
        is_irrel = qid in IRRELEVANT_IDS or not exp_pages

        t0 = time.time()
        r = run_retrieval(q["question"], cfg, backend)
        dt = time.time() - t0

        pages = r.selected_pages
        page_flags = [(doc_match(p["document_name"], exp_docs) and p["page_number"] in exp_pages) for p in pages]
        doc_flags = [doc_match(p["document_name"], exp_docs) for p in pages]

        rec = {
            "qid": qid, "answer_path": r.answer_path, "rerank_top": r.rerank_top_score,
            "n_pages": len(pages), "elapsed_s": round(dt, 2),
            "top3": [(p["document_name"][:18], p["page_number"], p["page_score"]) for p in pages[:3]],
        }

        if is_irrel:
            rejected = (r.answer_path == "none")
            irrel_reject.append(rejected)
            if r.rerank_top_score is not None:
                irrel_top_scores.append(r.rerank_top_score)
            rec["irrelevant"] = True
            rec["rejected"] = rejected
            print(f"  {qid:14s} [무관] rejected={rejected} rerank_top={r.rerank_top_score}")
        else:
            agg["page_hit@1"].append(bool(page_flags[:1] and page_flags[0]))
            agg["page_hit@3"].append(any(page_flags[:3]))
            agg["doc_hit@1"].append(bool(doc_flags[:1] and doc_flags[0]))
            agg["doc_hit@3"].append(any(doc_flags[:3]))
            if r.rerank_top_score is not None:
                rel_top_scores.append(r.rerank_top_score)
            rec.update({"page_hit@1": agg["page_hit@1"][-1], "page_hit@3": agg["page_hit@3"][-1]})
            print(f"  {qid:14s} p@1={int(agg['page_hit@1'][-1])} p@3={int(agg['page_hit@3'][-1])} "
                  f"path={r.answer_path} top={r.rerank_top_score} ({dt:.1f}s)")
        per_q.append(rec)

    def mean(xs):
        return round(sum(1.0 if x else 0.0 for x in xs) / len(xs), 4) if xs else 0.0

    summary = {k: mean(v) for k, v in agg.items()}
    summary["irrelevant_reject_rate"] = mean(irrel_reject)
    summary["n_relevant"] = len(agg["page_hit@1"])
    summary["n_irrelevant"] = len(irrel_reject)
    summary["rerank_score_dist"] = {
        "relevant_min": round(min(rel_top_scores), 3) if rel_top_scores else None,
        "relevant_p10": round(sorted(rel_top_scores)[len(rel_top_scores)//10], 3) if rel_top_scores else None,
        "irrelevant_max": round(max(irrel_top_scores), 3) if irrel_top_scores else None,
        "irrelevant_scores": [round(s, 3) for s in irrel_top_scores],
    }

    out = {"config": {"use_catalog_gate": cfg.use_catalog_gate, "rerank_floor": cfg.rerank_score_floor,
                       "candidates": cfg.retrieve_candidates, "final_pages": cfg.final_pages},
           "summary": summary, "per_question": per_q}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Phase 1 검색 품질 ===")
    print(f"  page_hit@1={summary['page_hit@1']}  page_hit@3={summary['page_hit@3']} (목표 >=0.75)")
    print(f"  doc_hit@1={summary['doc_hit@1']}  doc_hit@3={summary['doc_hit@3']} (목표 >=0.92)")
    print(f"  무관 거절={summary['irrelevant_reject_rate']} ({sum(irrel_reject)}/{len(irrel_reject)})")
    print(f"  리랭크 점수: 관련 min={summary['rerank_score_dist']['relevant_min']} "
          f"/ 무관 max={summary['rerank_score_dist']['irrelevant_max']} "
          f"-> floor 후보는 이 사이")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()

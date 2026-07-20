"""Phase 2 전체 평가: controller.answer_question를 36문항에 돌려 답변·검증·롤백 지표 측정.

핵심 지표(계획):
- answer_given_evidence_fail: 정답 페이지가 근거(selected_pages)에 있는데도 abstain -> 0 목표
- kw_hit: 최종 답변의 expected_answer_keywords 재현율 -> >=0.70 목표
- avg 모델 호출(임베딩 제외) -> <=3.5 목표
- 무관 거절, 롤백 발생/성공, confidence 분포, 지연
증분 저장(문항마다). 결과: test_3/probes/results/phase2_eval.json
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
from rag3.controller import answer_question  # noqa: E402

DATASETS = [
    ROOT / "rag_test" / "test_2" / "rag_eval_dataset.json",
    ROOT / "test12_total_test" / "test2_vlm_probe" / "vlm_probe_dataset.json",
]
OUT = Path(__file__).resolve().parent / "results" / "phase2_eval.json"
IRRELEVANT_IDS = {"irrelevant_001", "irrelevant_002", "irrelevant_003", "vi_001", "vi_002"}


def load_items():
    items = []
    for p in DATASETS:
        ds = json.loads(p.read_text(encoding="utf-8"))
        arr = ds if isinstance(ds, list) else list(ds.values())[0]
        items.extend(arr)
    return items


def doc_match(dn, exp_docs):
    return any(d == dn or Path(d).name == dn or d in dn or dn in d for d in exp_docs)


def norm(s):
    return "".join(str(s).split()).lower()


def kw_hit(answer, keywords):
    if not keywords:
        return None
    a = norm(answer)
    hit = sum(1 for k in keywords if norm(k) in a)
    return hit / len(keywords)


def main():
    cfg = load_config()
    backend = get_backend(cfg)
    print(f"enable_verify={cfg.enable_verify} enable_rollback={cfg.enable_rollback} "
          f"enable_crag={cfg.enable_crag} num_predict={cfg.ollama_num_predict}")
    items = load_items()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # resume: 기존 결과가 있으면 이어서(끝난 qid 건너뜀)
    results = []
    done = set()
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
            results = prev.get("results", [])
            done = {r["qid"] for r in results}
            if done:
                print(f"resume: 기존 {len(done)}문항 건너뜀")
        except Exception:
            results, done = [], set()

    for i, q in enumerate(items):
        qid = q["id"]
        if qid in done:
            continue
        exp_docs = q.get("expected_documents", [])
        exp_pages = q.get("expected_pages", [])
        kws = q.get("expected_answer_keywords", [])
        is_irrel = qid in IRRELEVANT_IDS or not exp_pages

        t0 = time.time()
        r = answer_question(q["question"], cfg, backend, run_id=qid)
        dt = time.time() - t0

        pages = r["selected_pages"]
        evidence_present = any(doc_match(p["document_name"], exp_docs) and p["page_number"] in exp_pages
                               for p in pages[:3])
        abstain = (r["answer_path"] == "none") or (r["confidence"] == "abstain")
        rec = {
            "qid": qid, "irrelevant": is_irrel,
            "answer_path": r["answer_path"], "confidence": r["confidence"],
            "evidence_present": evidence_present, "abstain": abstain,
            "kw_hit": kw_hit(r["final_answer"], kws),
            "model_calls": r["metrics"]["total_model_calls"],
            "verify_calls": r["metrics"].get("verify_calls", 0),
            "rollback": [h.get("action") for h in r.get("rollback_history", [])],
            "elapsed_s": round(dt, 1),
            "final_answer_head": (r["final_answer"] or "")[:120],
        }
        results.append(rec)
        OUT.write_text(json.dumps({"partial": True, "n_done": len(results), "results": results},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        tag = "무관" if is_irrel else f"ev={int(evidence_present)}"
        print(f"[{i+1}/{len(items)}] {qid:14s} {tag} path={r['answer_path']} conf={r['confidence']} "
              f"kw={rec['kw_hit']} calls={rec['model_calls']} rb={rec['rollback']} ({dt:.0f}s)")

    # 집계
    rel = [r for r in results if not r["irrelevant"]]
    irr = [r for r in results if r["irrelevant"]]
    ev_present = [r for r in rel if r["evidence_present"]]
    given_evidence_fail = [r for r in ev_present if r["abstain"]]
    answered = [r for r in rel if not r["abstain"]]
    kw_vals = [r["kw_hit"] for r in rel if r["kw_hit"] is not None]
    rolled = [r for r in results if r["rollback"]]

    summary = {
        "n_relevant": len(rel), "n_irrelevant": len(irr),
        "evidence_present_count": len(ev_present),
        "answer_given_evidence_fail": len(given_evidence_fail),
        "answer_given_evidence_fail_ids": [r["qid"] for r in given_evidence_fail],
        "answered_rate": round(len(answered) / len(rel), 4) if rel else 0,
        "avg_kw_hit": round(sum(kw_vals) / len(kw_vals), 4) if kw_vals else 0,
        "irrelevant_reject_rate": round(sum(1 for r in irr if r["abstain"] or r["answer_path"] == "none") / len(irr), 4) if irr else 0,
        "avg_model_calls": round(sum(r["model_calls"] for r in rel) / len(rel), 2) if rel else 0,
        "rollback_count": len(rolled),
        "rollback_ids": [r["qid"] for r in rolled],
        "confidence_dist": {c: sum(1 for r in results if r["confidence"] == c)
                            for c in ("high", "low", "abstain", "unknown")},
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in results) / len(results), 1),
        "max_elapsed_s": round(max(r["elapsed_s"] for r in results), 1),
    }
    OUT.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Phase 2 요약 ===")
    print(f"  근거있는데실패: {summary['answer_given_evidence_fail']}/{summary['evidence_present_count']} "
          f"{summary['answer_given_evidence_fail_ids']} (목표 0)")
    print(f"  답변율(관련): {summary['answered_rate']}  kw_hit: {summary['avg_kw_hit']} (목표>=0.70)")
    print(f"  무관 거절: {summary['irrelevant_reject_rate']}")
    print(f"  avg 모델호출: {summary['avg_model_calls']} (목표<=3.5)  롤백: {summary['rollback_count']}건 {summary['rollback_ids']}")
    print(f"  confidence: {summary['confidence_dist']}")
    print(f"  지연 avg {summary['avg_elapsed_s']}s / max {summary['max_elapsed_s']}s")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()

"""P0-D: 리랭커(bge-reranker-v2-m3) 선검증 — 기존 page_index를 그대로 쓰고 재정렬 효과만 측정.

목적(계획 문제 2/4/5): 정답이 2~3순위로 밀려 실패하는 문제를, top-K 후보를 리랭커로 재정렬하면
1순위로 끌어올릴 수 있는지 파이프라인 수정 전에 확인한다.

방법:
- 36문항(rag_eval 16 + vlm_probe 20)에 대해 page_index를 GLOBAL(문서필터 없음) top-K 질의
- baseline = RRF 순서, reranked = bge-reranker-v2-m3 (question, page_text) 점수 내림차순
- page_hit@1/@3, doc_hit@1/@3, MRR 을 baseline vs reranked 로 비교
- 리랭크 지연(20쌍) 측정
정답 판정: 후보의 document_name ∈ expected_documents AND page_number ∈ expected_pages

결과: test_3/probes/results/p0d_rerank.json
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

ROOT = Path(__file__).resolve().parents[2]  # 챗봇/
sys.path.insert(0, str(ROOT / "test_3"))
from rag3.config import load_config  # noqa: E402
from rag3.models import get_backend  # noqa: E402
from rag3.index import get_index  # noqa: E402
from rag3 import metrics  # noqa: E402

TEST2 = ROOT / "rag_test" / "test_2" / "rag2"
DATASETS = [
    ROOT / "rag_test" / "test_2" / "rag_eval_dataset.json",
    ROOT / "test12_total_test" / "test2_vlm_probe" / "vlm_probe_dataset.json",
]
OUT = Path(__file__).resolve().parent / "results" / "p0d_rerank.json"

TOPK = 20            # 후보 풀 크기
RERANK_MAX_LEN = 2048  # 리랭커 입력 최대 토큰(표가 페이지 중간일 수 있어 넉넉히)
RERANKER = "BAAI/bge-reranker-v2-m3"


def load_items() -> list[dict]:
    items = []
    for p in DATASETS:
        ds = json.loads(p.read_text(encoding="utf-8"))
        arr = ds if isinstance(ds, list) else list(ds.values())[0]
        for q in arr:
            items.append(q)
    return items


def is_hit(meta: dict, exp_docs: list[str], exp_pages: list[int]) -> bool:
    dn = meta.get("document_name", "")
    pn = meta.get("page_number")
    doc_ok = any(d == dn or Path(d).name == dn or d in dn or dn in d for d in exp_docs)
    return doc_ok and (pn in exp_pages)


def doc_hit(meta: dict, exp_docs: list[str]) -> bool:
    dn = meta.get("document_name", "")
    return any(d == dn or Path(d).name == dn or d in dn or dn in d for d in exp_docs)


def metrics_from(order: list[dict], exp_docs, exp_pages) -> dict:
    """order = 후보 메타 리스트(정렬됨). page_hit@1/@3, doc_hit@3, page MRR."""
    page_flags = [is_hit(m, exp_docs, exp_pages) for m in order]
    doc_flags = [doc_hit(m, exp_docs) for m in order]
    mrr = 0.0
    for i, f in enumerate(page_flags):
        if f:
            mrr = 1.0 / (i + 1)
            break
    return {
        "page_hit@1": bool(page_flags[:1] and page_flags[0]),
        "page_hit@3": any(page_flags[:3]),
        "doc_hit@1": bool(doc_flags[:1] and doc_flags[0]),
        "doc_hit@3": any(doc_flags[:3]),
        "page_mrr": round(mrr, 4),
    }


def main() -> None:
    cfg = load_config(overrides={
        "index_dir": str(TEST2 / "index"),
        "cache_dir": str(TEST2 / "cache"),
    })
    print(f"index_dir={cfg.index_dir}  chroma={cfg.chroma_dir}")
    backend = get_backend(cfg)
    page_index = get_index("page_index", cfg, backend)
    cnt = page_index.count()
    print(f"page_index count = {cnt}")
    if cnt == 0:
        print("!! page_index 비어있음 — test_2 인덱스 경로 확인 필요")
        return

    print(f"리랭커 로드: {RERANKER} ...")
    from sentence_transformers import CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANKER, max_length=RERANK_MAX_LEN, device=device)

    items = load_items()
    print(f"문항 수: {len(items)}")

    agg = {"baseline": [], "reranked": []}
    per_q = []
    rerank_times = []

    for q in items:
        qid = q["id"]
        question = q["question"]
        exp_docs = q.get("expected_documents", [])
        exp_pages = q.get("expected_pages", [])

        emb = backend.embed([question], is_query=True)[0]
        metrics.record_embed()
        cands = page_index.query(question, n_results=TOPK, query_embedding=emb)  # GLOBAL
        base_order = [c.metadata for c in cands]

        # 리랭크
        pairs = [(question, c.text[:6000]) for c in cands]
        t0 = time.time()
        scores = reranker.predict(pairs) if pairs else []
        rerank_times.append(time.time() - t0)
        reranked = sorted(zip(cands, scores), key=lambda x: -float(x[1]))
        rr_order = [c.metadata for c, _ in reranked]

        m_base = metrics_from(base_order, exp_docs, exp_pages)
        m_rr = metrics_from(rr_order, exp_docs, exp_pages)
        agg["baseline"].append(m_base)
        agg["reranked"].append(m_rr)
        per_q.append({
            "qid": qid, "type": q.get("question_type"), "exp_pages": exp_pages,
            "baseline": m_base, "reranked": m_rr,
            "base_top3": [(m.get("document_name", "")[:20], m.get("page_number")) for m in base_order[:3]],
            "rr_top3": [(m.get("document_name", "")[:20], m.get("page_number")) for m in rr_order[:3]],
        })
        flag = ""
        if not m_base["page_hit@1"] and m_rr["page_hit@1"]:
            flag = "  <- 리랭크로 1순위 승격"
        elif m_base["page_hit@1"] and not m_rr["page_hit@1"]:
            flag = "  <- 리랭크가 1순위 훼손"
        print(f"  {qid:10s} base@1={int(m_base['page_hit@1'])} @3={int(m_base['page_hit@3'])} | "
              f"rr@1={int(m_rr['page_hit@1'])} @3={int(m_rr['page_hit@3'])}{flag}")

    def mean(key_group, key):
        vals = [1.0 if x[key] is True else (0.0 if x[key] is False else x[key]) for x in agg[key_group]]
        return round(sum(vals) / len(vals), 4)

    summary = {}
    for grp in ("baseline", "reranked"):
        summary[grp] = {k: mean(grp, k) for k in ("page_hit@1", "page_hit@3", "doc_hit@1", "doc_hit@3", "page_mrr")}

    out = {
        "reranker": RERANKER, "topk": TOPK, "n": len(items),
        "page_index_count": cnt,
        "rerank_latency_s": {"mean": round(sum(rerank_times) / len(rerank_times), 3),
                              "max": round(max(rerank_times), 3)},
        "summary": summary,
        "per_question": per_q,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== P0-D 요약 ===")
    print(f"  baseline : page_hit@1={summary['baseline']['page_hit@1']} @3={summary['baseline']['page_hit@3']} mrr={summary['baseline']['page_mrr']}")
    print(f"  reranked : page_hit@1={summary['reranked']['page_hit@1']} @3={summary['reranked']['page_hit@3']} mrr={summary['reranked']['page_mrr']}")
    d1 = summary['reranked']['page_hit@3'] - summary['baseline']['page_hit@3']
    print(f"  page_hit@3 개선: {d1:+.4f} (통과기준 +0.10)")
    print(f"  리랭크 지연(20쌍): 평균 {out['rerank_latency_s']['mean']}s / 최대 {out['rerank_latency_s']['max']}s")
    print(f"\n결과 저장: {OUT}")


if __name__ == "__main__":
    main()

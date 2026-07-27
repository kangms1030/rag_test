# -*- coding: utf-8 -*-
"""FAQ 의미 매칭의 자동채택/되묻기 임계값을 실측으로 정한다 (작업 3).

문자 유사도(fuzz.ratio)와 의미 유사도는 **점수 스케일이 완전히 다르다**. 현행 임계
0.90/0.75 는 fuzz 스케일 값이므로 의미 매칭에 그대로 쓰면 안 된다.

측정 방법
    양성(같은 FAQ 를 가리켜야 하는 쌍):
        - FAQ 질문 자기 자신 (상한 확인)
        - eval_set.json 의 faq_paraphrase 문항 ↔ expected_faq_id (사람이 만든 진짜 패러프레이즈)
    음성(서로 다른 FAQ 를 가리켜야 하는 쌍):
        - FAQ 질문 ↔ 자신이 아닌 최고 점수 FAQ (가장 헷갈리는 오답 = hard negative)
        - eval_set.json 의 out_of_scope 문항 ↔ 최고 점수 FAQ

    두 분포가 겹치는 구간이 "되묻기(clarify) 구간"이다.
        자동채택 임계 = 음성 최고점보다 위 (오답 채택 0을 목표)
        되묻기 하한   = 양성 최저점보다 아래 (정답을 놓치지 않는 것을 목표)

실행:
    <intern_chatbot python> -X utf8 chatbot_demo_v2/scripts/calibrate_faq_threshold.py
    ... --scorer embed        # 임베딩 코사인만 (리랭커 생략, 빠름)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT.parent))

from chatbot_demo_v2.config.settings import load_settings             # noqa: E402
from chatbot_demo_v2.rag.adapter_util import prepare_ragcore_imports  # noqa: E402


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)


def pct(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="FAQ 매칭 임계값 캘리브레이션")
    ap.add_argument("--scorer", choices=["embed", "rerank"], default="rerank",
                    help="embed=임베딩 코사인만 / rerank=임베딩 top-10 후 크로스인코더 재점수(운영과 동일)")
    ap.add_argument("--topk", type=int, default=10, help="rerank 모드에서 재점수할 후보 수")
    ap.add_argument("--out", default=str(PKG_ROOT / "runtime" / "reports" / "faq_threshold_calibration.json"))
    args = ap.parse_args()

    settings = load_settings()
    prepare_ragcore_imports(settings)
    from rag3.config import load_config
    from rag3.models import OllamaBackend

    config = load_config(str(settings.ragcore_config))
    backend = OllamaBackend(config)

    emb_path = settings.data_dir / "faq_embeddings.json"
    if not emb_path.is_file():
        print("먼저 build_faq_embeddings.py 를 실행하세요:", emb_path)
        return 1
    fe = json.loads(emb_path.read_text(encoding="utf-8"))
    ids = fe["ids"]
    questions = fe["questions"]
    M = _l2(np.asarray(fe["vectors"], dtype=np.float32))
    id2idx = {q: i for i, q in enumerate(ids)}
    print("FAQ 임베딩 %d건 (dim=%d, %s)" % (len(ids), fe["dim"], fe["embedding_model"]))

    reranker = None
    if args.scorer == "rerank":
        from rag3.rerank import get_reranker
        reranker = get_reranker(config)
        print("리랭커: %s" % config.rerank_model)

    def score_all(query: str) -> np.ndarray:
        """질의 하나에 대한 전체 FAQ 점수 벡터."""
        q = _l2(np.asarray(backend.embed([query], is_query=True)[0], dtype=np.float32))
        cos = M @ q
        if reranker is None:
            return cos
        top = np.argsort(-cos)[: args.topk]
        hits = reranker.rank(query, [questions[i] for i in top])
        out = np.full(len(ids), -1.0, dtype=np.float32)
        for h in hits:
            out[top[h.index]] = h.score
        return out

    # ---- 표본 수집 ----
    evalset = json.loads((settings.data_dir / "eval_set.json").read_text(encoding="utf-8"))
    para = [it for it in evalset["items"] if it["category"] == "faq_paraphrase" and it.get("expected_faq_id")]
    oos = [it for it in evalset["items"] if it["category"] == "out_of_scope"]
    ambig = [it for it in evalset["items"] if it["category"] == "ambiguous"]

    rows = {"self": [], "paraphrase": [], "hard_negative": [], "out_of_scope": [], "ambiguous": []}

    print("\n[1/4] 자기 자신 (양성 상한) — %d건" % len(ids))
    for i, q in enumerate(questions):
        s = score_all(q)
        rows["self"].append(float(s[i]))
        # 자기 자신을 제외한 최고점 = hard negative
        s2 = s.copy(); s2[i] = -1e9
        rows["hard_negative"].append(float(s2.argmax() and s2.max() or s2.max()))
        if (i + 1) % 50 == 0:
            print("   %d/%d" % (i + 1, len(ids)), flush=True)

    print("[2/4] 사람이 만든 패러프레이즈 (양성 실제) — %d건" % len(para))
    para_detail = []
    for it in para:
        s = score_all(it["question"])
        tgt = id2idx.get(it["expected_faq_id"])
        best = int(np.argmax(s))
        rows["paraphrase"].append(float(s[tgt]) if tgt is not None else float("nan"))
        para_detail.append({
            "question": it["question"], "expected": it["expected_faq_id"],
            "expected_score": float(s[tgt]) if tgt is not None else None,
            "best_id": ids[best], "best_score": float(s[best]),
            "rank_of_expected": int((s > s[tgt]).sum() + 1) if tgt is not None else None,
        })

    print("[3/4] 범위 밖 질문 (음성) — %d건" % len(oos))
    for it in oos:
        rows["out_of_scope"].append(float(np.max(score_all(it["question"]))))

    print("[4/4] 모호 질문 (되묻기 대상) — %d건" % len(ambig))
    ambig_detail = []
    for it in ambig:
        s = score_all(it["question"])
        order = np.argsort(-s)[:3]
        rows["ambiguous"].append(float(s[order[0]]))
        ambig_detail.append({"question": it["question"],
                             "top3": [{"id": ids[i], "score": float(s[i])} for i in order]})

    # ---- 분포 요약 ----
    print("\n" + "=" * 92)
    print("%-14s %5s %8s %8s %8s %8s %8s" % ("표본", "n", "최소", "5%", "중앙", "95%", "최대"))
    print("-" * 92)
    for k in ["self", "paraphrase", "hard_negative", "out_of_scope", "ambiguous"]:
        v = [x for x in rows[k] if not np.isnan(x)]
        if not v:
            continue
        print("%-14s %5d %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (k, len(v), min(v), pct(v, 5), pct(v, 50), pct(v, 95), max(v)))
    print("=" * 92)

    pos = [x for x in rows["paraphrase"] if not np.isnan(x)]
    neg = rows["hard_negative"] + rows["out_of_scope"]

    # 자동채택 임계: 음성 95분위 위 (오채택 5% 이하), 되묻기 하한: 양성 5분위 아래
    accept = round(max(pct(neg, 95), pct(pos, 25)), 3)
    clarify = round(min(pct(pos, 5), pct(neg, 75)), 3)
    if clarify >= accept:
        clarify = round(accept * 0.6, 3)

    print("\n권고 임계값 (scorer=%s)" % args.scorer)
    print("  SCENARIO_MATCH_THRESHOLD (자동채택) = %.3f" % accept)
    print("  CLARIFY_MIN_SCORE        (되묻기)   = %.3f" % clarify)
    print("  → best >= %.3f 이면 FAQ 즉답 / %.3f <= best < %.3f 이면 되묻기 / 그 아래는 RAG"
          % (accept, clarify, accept))
    tp = sum(1 for x in pos if x >= accept)
    fp = sum(1 for x in neg if x >= accept)
    cl = sum(1 for x in pos if clarify <= x < accept)
    print("  이 임계에서 패러프레이즈 %d/%d 자동채택, %d건 되묻기, 오채택(음성) %d/%d"
          % (tp, len(pos), cl, fp, len(neg)))

    print("\n패러프레이즈 문항별:")
    for d in para_detail:
        ok = "OK " if d["rank_of_expected"] == 1 else "MISS"
        print("  %s %-40s 기대%s=%.3f(%d위) 최고%s=%.3f"
              % (ok, d["question"][:38], d["expected"], d["expected_score"] or 0,
                 d["rank_of_expected"] or 0, d["best_id"], d["best_score"]))

    print("\n모호 문항 top-3 (되묻기 후보로 쓸 만한지):")
    for d in ambig_detail:
        print("  %-34s %s" % (d["question"][:32],
                              " | ".join("%s %.3f" % (t["id"], t["score"]) for t in d["top3"])))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "scorer": args.scorer, "topk": args.topk,
        "recommend": {"scenario_match_threshold": accept, "clarify_min_score": clarify},
        "distributions": rows, "paraphrase_detail": para_detail, "ambiguous_detail": ambig_detail,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n저장:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

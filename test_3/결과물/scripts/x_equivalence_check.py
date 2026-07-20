"""Phase 6.0 등가성 게이트: rag3x(전 플래그 OFF) == rag3 동작 증명 + 골든 회귀.

1) base config 등가: load_x_config가 rag3 원본 필드를 하나도 바꾸지 않았는지.
2) 검색 결정론 등가: 동일 질문에 rag3.run_retrieval == rag3x 경로 검색결과(선정페이지/점수).
3) 골든 회귀: controller_x 4문항 kw_hit이 varfix 기준치 이상 + 무관 거절(원본 verify와 동일 기준).
"""
import dataclasses
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
TEST3 = Path(__file__).resolve().parents[1]
ROOT = TEST3.parent
sys.path.insert(0, str(TEST3))

from rag3.config import load_config
from rag3.retrieve import run_retrieval
from rag3.models import get_backend
from rag3x.xconfig import load_x_config
from rag3x.controller_x import answer_question as x_answer_question

QIDS = ["core_001", "core_004", "vp_009", "irrelevant_001"]
DATASETS = [ROOT / "rag_test" / "test_2" / "rag_eval_dataset.json",
            ROOT / "test12_total_test" / "test2_vlm_probe" / "vlm_probe_dataset.json"]

items = {}
for p in DATASETS:
    ds = json.loads(p.read_text(encoding="utf-8"))
    arr = ds if isinstance(ds, list) else list(ds.values())[0]
    for q in arr:
        items[q["id"]] = q

base = {r["qid"]: r for r in json.loads(
    (TEST3 / "probes" / "results" / "varfix_eval.json").read_text(encoding="utf-8"))["results"]}


def norm(s):
    return "".join(str(s).split()).lower()


def kw_hit(ans, kws):
    if not kws:
        return None
    a = norm(ans)
    return sum(1 for k in kws if norm(k) in a) / len(kws)


fails = []

# --- (1) base config 등가 ---
cfg = load_config()
xcfg = load_x_config()
base_fields = [f.name for f in dataclasses.fields(cfg)]
diffs = [f for f in base_fields if getattr(cfg, f) != getattr(xcfg, f)]
if diffs:
    fails.append(f"base config 필드 변경됨: {diffs}")
print(f"[1] base config 등가: {'OK' if not diffs else 'FAIL ' + str(diffs)} "
      f"(원본 {len(base_fields)}필드 + 실험플래그 {len(vars(xcfg)) - len(base_fields)}개 부착)")

backend = get_backend(xcfg)

# --- (2) 검색 결정론 등가 (LLM 없는 구간, 완전 재현) ---
for qid in QIDS:
    q = items[qid]
    r1 = run_retrieval(q["question"], cfg, backend)
    r2 = run_retrieval(q["question"], xcfg, backend)
    sig1 = [(p["document_name"], p["page_number"], p["page_score"]) for p in r1.selected_pages]
    sig2 = [(p["document_name"], p["page_number"], p["page_score"]) for p in r2.selected_pages]
    ok = sig1 == sig2 and r1.answer_path == r2.answer_path and r1.rerank_top_score == r2.rerank_top_score
    if not ok:
        fails.append(f"{qid}: 검색 등가 실패 {sig1} != {sig2}")
    print(f"[2] {qid:16s} 검색등가 {'OK' if ok else 'FAIL'} top={r2.rerank_top_score} path={r2.answer_path}")

# --- (3) 골든 회귀 (controller_x, 플래그 OFF) ---
for qid in QIDS:
    q = items[qid]
    t0 = time.time()
    r = x_answer_question(q["question"], xcfg, backend, run_id=qid)
    dt = time.time() - t0
    kh = kw_hit(r["final_answer"], q.get("expected_answer_keywords", []))
    bk = base.get(qid, {}).get("kw_hit")
    print(f"[3] {qid:16s} path={r['answer_path']:6s} conf={r['confidence']:8s} "
          f"kw={kh} (기준 {bk}) {dt:.0f}s :: {(r['final_answer'] or '')[:55].replace(chr(10),' ')}")
    if qid.startswith("irrelevant"):
        if not (r["answer_path"] == "none" or r["confidence"] == "abstain"):
            fails.append(f"{qid}: 무관 질문 거절 실패")
    elif kh is not None and bk is not None and kh < bk - 1e-9:
        fails.append(f"{qid}: kw {bk} -> {kh} 회귀")

if fails:
    print("\nEQUIVALENCE FAIL:", fails)
    sys.exit(1)
print("\nRAG3X EQUIVALENCE OK — 전 플래그 OFF에서 rag3와 동작 등가 + 골든 회귀 통과")

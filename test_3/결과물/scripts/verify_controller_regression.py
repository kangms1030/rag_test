"""controller 경로 회귀 체크(대표 4문항) — FINAL 기준치(varfix_eval.json)와 kw_hit 비교."""
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
from rag3.controller import answer_question
from rag3.models import get_backend

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


cfg = load_config()
backend = get_backend(cfg)
fails = []
for qid in QIDS:
    q = items[qid]
    t0 = time.time()
    r = answer_question(q["question"], cfg, backend, run_id=qid)
    dt = time.time() - t0
    kh = kw_hit(r["final_answer"], q.get("expected_answer_keywords", []))
    b = base.get(qid, {})
    bk = b.get("kw_hit")
    print(f"{qid:16s} path={r['answer_path']:6s} conf={r['confidence']:8s} "
          f"kw={kh} (기준 {bk}) {dt:.0f}s :: {(r['final_answer'] or '')[:60].replace(chr(10),' ')}")
    if qid.startswith("irrelevant"):
        if not (r["answer_path"] == "none" or r["confidence"] == "abstain"):
            fails.append(f"{qid}: 무관 질문이 거절되지 않음")
    elif kh is not None and bk is not None and kh < bk - 1e-9:
        fails.append(f"{qid}: kw {bk} -> {kh} 회귀")

if fails:
    print("REGRESSION FAIL:", fails)
    sys.exit(1)
print("CONTROLLER REGRESSION OK (기준치 이상)")

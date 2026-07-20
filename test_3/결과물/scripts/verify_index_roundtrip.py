"""3(a) 검증: 인덱스 복사본에서 remove_doc→append 왕복 후 무결성/검색 동등성 확인.

프로덕션 인덱스는 건드리지 않는다(tmp/index_copy에 복사해 작업).
"""
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
TEST3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST3))

from rag3.config import load_config
from rag3.flat_index import FlatChunkIndex
from rag3.models import get_backend

SRC_INDEX = TEST3 / "rag3" / "index"
COPY = TEST3 / "tmp" / "index_copy"

if COPY.exists():
    shutil.rmtree(COPY)
(COPY / "flat_chunk").parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(SRC_INDEX / "flat_chunk", COPY / "flat_chunk")
shutil.copy2(SRC_INDEX / "page_store.json", COPY / "page_store.json")

config = load_config(None, {"index_dir": str(COPY), "output_dir": str(TEST3 / "tmp" / "out_copy")})
backend = get_backend(config)
flat = FlatChunkIndex(config, backend)

n0 = flat.count()
print("초기 청크 수:", n0)
assert n0 == 2562, f"예상 2562, 실제 {n0}"

# 대상: 가장 작은 문서의 slug (자가진단 체크리스트)
data = json.loads((COPY / "flat_chunk" / backend.backend_id / "docs.json").read_text(encoding="utf-8"))
by_slug = {}
for i, m in enumerate(data["metas"]):
    by_slug.setdefault(m["doc_slug"], []).append(i)
slug = min(by_slug, key=lambda s: len(by_slug[s]))
idxs = by_slug[slug]
print(f"대상 slug={slug} 청크 {len(idxs)}개")
saved = ([data["ids"][i] for i in idxs], [data["docs"][i] for i in idxs], [data["metas"][i] for i in idxs])

q = "학교 무선인터넷 자가진단 체크리스트 항목"
before = flat.query(q, n_results=10)
before_ids = [h.id for h in before]

# 왕복
removed = flat.remove_doc(slug)
assert removed == len(idxs), f"제거 수 불일치: {removed} != {len(idxs)}"
assert flat.count() == n0 - removed
# 제거 후 해당 slug가 결과에 안 나옴
mid = flat.query(q, n_results=10)
assert all(h.metadata["doc_slug"] != slug for h in mid), "제거 후에도 해당 문서가 검색됨"

added = flat.append(*saved)
assert added == len(idxs)
assert flat.count() == n0, f"복원 후 count {flat.count()} != {n0}"

# npz 포맷/무결성: ids dtype object, emb float32, 정규화 유지
with np.load(COPY / "flat_chunk" / backend.backend_id / "vectors.npz", allow_pickle=True) as npz:
    emb = npz["emb"]
    assert emb.dtype == np.float32, emb.dtype
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"정규화 깨짐: {norms.min()}..{norms.max()}"
    assert emb.shape[0] == n0

after = flat.query(q, n_results=10)
after_ids = [h.id for h in after]
overlap = len(set(before_ids) & set(after_ids))
print("검색 top10 왕복 전:", before_ids[:5])
print("검색 top10 왕복 후:", after_ids[:5])
print(f"top10 교집합: {overlap}/10")
assert overlap >= 9, "왕복 후 검색 결과가 크게 달라짐(재임베딩/순서 문제 의심)"

# 중복 append 방어
try:
    flat.append(saved[0][:1], saved[1][:1], saved[2][:1])
    raise AssertionError("중복 id append가 막히지 않음")
except ValueError as e:
    print("중복 방어 OK:", str(e)[:60])

shutil.rmtree(COPY)
print("ROUNDTRIP OK")

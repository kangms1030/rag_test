"""3(b) 검증: collect_chunk_records(추출된 헬퍼)가 프로덕션 인덱스(docs.json)와
동일한 id/색인텍스트/메타를 생성하는지 13개 문서 전체에서 비교. 임베딩/모델 호출 없음."""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
TEST3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST3))

from rag3.catalog import load_catalog, match_catalog_to_pdfs
from rag3.config import load_config
from rag3.ingest import _catalog_prefix_map, _load_source_manifest, collect_chunk_records
from rag3.utils import doc_slug

config = load_config()
rows = load_catalog(config)
match_catalog_to_pdfs(rows, config.documents_dir)
prefix_map = _catalog_prefix_map(rows)

prod = json.loads(
    (config.index_dir / "flat_chunk" / "ollama-embeddinggemma" / "docs.json").read_text(encoding="utf-8"))
prod_text = dict(zip(prod["ids"], prod["docs"]))
prod_meta = dict(zip(prod["ids"], prod["metas"]))

total = 0
mismatch = 0
regen_ids = set()
for row in rows:
    if not row.matched_file_path:
        continue
    slug = doc_slug(row.matched_file_path)
    doc_info = _load_source_manifest(config, slug)
    assert doc_info is not None, f"{slug} manifest 없음"
    rec = collect_chunk_records(config, prefix_map, slug, doc_info)
    assert rec is not None, f"{slug} content_list 없음"
    ids, texts, metas, _ = rec
    regen_ids.update(ids)
    for cid, txt, meta in zip(ids, texts, metas):
        total += 1
        if cid not in prod_text:
            print(f"[신규 id] {cid}")
            mismatch += 1
        elif prod_text[cid] != txt:
            print(f"[텍스트 불일치] {cid}")
            mismatch += 1
        elif prod_meta[cid] != meta:
            print(f"[메타 불일치] {cid}: {set(prod_meta[cid].items()) ^ set(meta.items())}")
            mismatch += 1

missing = set(prod_text) - regen_ids
print(f"재생성 청크 {total}개, 프로덕션 {len(prod_text)}개, 불일치 {mismatch}, 프로덕션에만 있는 id {len(missing)}")
assert mismatch == 0 and not missing and total == len(prod_text)
print("INGEST EQUIVALENCE OK (byte-identical)")

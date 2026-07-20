"""코퍼스 문서별 페이지 텍스트를 UTF-8 파일로 덤프(종합형 평가셋 초안용). 셸 인코딩 우회."""
import json
from collections import defaultdict
from pathlib import Path

ps = json.load(open("rag3/index/page_store.json", encoding="utf-8"))
byname = defaultdict(list)
for k, v in ps.items():
    m = v.get("meta", {})
    byname[m.get("document_name", "?")].append((m.get("page_number"), v.get("text", "")))

# 문서명 목록 + 페이지수
names = sorted(byname.keys(), key=lambda n: -len(byname[n]))
idx = ["=== 문서 목록 ==="]
for i, n in enumerate(names):
    idx.append(f"[{i}] {len(byname[n])}p  {n}")
Path("tmp/corpus_index.txt").write_text("\n".join(idx), encoding="utf-8")

# 전 문서 페이지 텍스트(앞 700자)를 문서별 파일로
for i, n in enumerate(names):
    pages = sorted(byname[n])
    lines = [f"###### [{i}] {n} ({len(pages)}p) ######"]
    for pn, txt in pages:
        t = (txt or "").strip().replace("\n", " ")
        if len(t) > 30:
            lines.append(f"p{pn}: {t[:700]}")
    Path(f"tmp/corpus_doc_{i:02d}.txt").write_text("\n\n".join(lines), encoding="utf-8")
print(f"{len(names)}개 문서 덤프 → tmp/corpus_index.txt, corpus_doc_00..{len(names)-1:02d}.txt")

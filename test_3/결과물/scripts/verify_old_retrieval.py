"""신규 문서가 있는 상태에서 기존 문서(core_001) 검색이 그대로인지 (LLM 호출 없음)."""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
TEST3 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST3))

from rag3.config import load_config
from rag3.models import get_backend
from rag3.retrieve import run_retrieval

config = load_config()
backend = get_backend(config)
q = (TEST3 / "tmp" / "q_core001.txt").read_text(encoding="utf-8").strip()
r = run_retrieval(q, config, backend)
pages = [(p["document_name"], p["page_number"], round(p["page_score"], 4)) for p in r.selected_pages]
for p in pages:
    print(p)
assert pages[0][0] == "(첨부)_5단계_스쿨넷서비스_제공_가이드.pdf" and pages[0][1] == 11, "core_001 top1 회귀"
assert [x[1] for x in pages] == [11, 5, 14], f"페이지 순서 변화: {pages}"
print("OLD RETRIEVAL OK (p11/p5/p14 동일)")

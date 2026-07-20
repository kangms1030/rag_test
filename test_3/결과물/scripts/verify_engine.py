"""1단계 검증: Rag3Engine.ask가 베이스라인과 같은 답을 내고 근거 이미지가 해석/복사되는지."""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag3 import Rag3Engine

q = Path(__file__).with_name("q_core001.txt").read_text(encoding="utf-8").strip()
engine = Rag3Engine()
print("[health]", engine.health())
r = engine.ask(q, save_evidence=True)

print("[path]", r["answer_path"], "[conf]", r["confidence"])
print("[answer]", r["final_answer"][:200].replace("\n", " "))
ok = True
for ev in r["evidence"]:
    resolved = ev.get("page_image_resolved")
    exists = bool(resolved) and Path(resolved).exists()
    ok &= exists
    print(f"  ev p{ev['page_number']:>3} resolved={resolved} exists={exists}")
for f in r.get("evidence_files", []):
    exists = Path(f["file"]).exists()
    ok &= exists
    print(f"  file {f['file']} exists={exists}")
assert r["answer_path"] == "text" and r["confidence"] == "high", "베이스라인과 경로/신뢰도 불일치"
assert "2,907,000" in r["final_answer"] and "4,121,000" in r["final_answer"], "핵심 숫자 누락"
assert ok, "이미지 경로 해석/복사 실패"
print("VERIFY OK")

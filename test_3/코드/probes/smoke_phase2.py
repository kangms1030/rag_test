"""Phase 2 컨트롤러 스모크 테스트: 대표 3문항(정상 text / 과거 실패 / 무관)."""
import io, sys, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, "test_3")
from rag3.config import load_config
from rag3.models import get_backend
from rag3.controller import answer_question

cfg = load_config()
backend = get_backend(cfg)

qs = [
    ("core_001", "스쿨넷 5단계 재해복구(DR)서비스 요금체계에서 제주특별자치도교육청의 100M와 500M 월 요금은 각각 얼마이며, 해당 요금이 책정된 배경은 무엇인가요?"),
    ("vp_006", "MDM 매뉴얼의 대여 이력 관리 화면에서 처리자 'jhkim2761'이 대여 처리한 단말의 시리얼번호와 반납예정일은 무엇이며, 화면에 표시된 전체 이력 건수는 몇 건인가요?"),
    ("vi_001", "인공지능으로 주식 자동매매 프로그램을 만드는 방법을 알려주세요."),
]

for qid, q in qs:
    print(f"\n===== {qid} =====")
    r = answer_question(q, cfg, backend, run_id=qid)
    print("path:", r["answer_path"], "| confidence:", r["confidence"])
    print("answer:", (r["final_answer"] or "")[:200])
    print("verify:", r.get("verification"))
    print("rollback:", r.get("rollback_history"))
    print("calls:", r["metrics"]["total_model_calls"], "timings:", r["metrics"]["timings_seconds"])

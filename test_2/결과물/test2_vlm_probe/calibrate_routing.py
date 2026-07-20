"""vlm_probe_dataset.json 20문항의 라우팅 사전점검 (검색만 수행, 답변 LLM 미호출).

목적: 20문항 중 vision을 기대하는 15문항이 실제로 rag2._route()에서 "vision"으로
판정되는지, text 기대 3문항이 "text"로, 무관 2문항이 "none"으로 판정되는지 저비용으로
먼저 확인한다. run_retrieval/retrieve_no_catalog는 answer 단계(LLM 호출) 이전까지만
실행하므로 embed 호출만 발생하고 gemma4:12b 텍스트/비전 생성 호출은 없다.

rag2 소스는 수정하지 않고 import만 한다(기존 test_2_timecost/test12_total_test 계약과 동일).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TEST2_DIR = _HERE.parent.parent / "test_2"
_TEST2_TIMECOST_DIR = _HERE.parent.parent / "test_2_timecost"
sys.path.insert(0, str(_TEST2_DIR))
sys.path.insert(0, str(_TEST2_TIMECOST_DIR))

from rag2.config import load_config  # noqa: E402
from rag2.models import get_backend  # noqa: E402
from rag2.retrieve import run_retrieval  # noqa: E402

from retrieve_no_catalog import retrieve_no_catalog  # noqa: E402

DATASET = _HERE / "vlm_probe_dataset.json"


def main() -> None:
    items = json.loads(DATASET.read_text(encoding="utf-8"))
    config = load_config()
    backend = get_backend(config)

    rows = []
    for item in items:
        if item["question_type"] == "irrelevant":
            expected_path = "none"
        else:
            expected_path = item["expected_answer_path"]
        expected_pages = set(item.get("expected_pages", []))

        for mode, fn in (("catalog", run_retrieval), ("no_catalog", retrieve_no_catalog)):
            retrieval = fn(item["question"], config, backend)
            selected_pages = [p["page_number"] for p in retrieval.selected_pages]
            top1_page = selected_pages[0] if selected_pages else None
            page_hit_top1 = top1_page in expected_pages if expected_pages else None
            page_hit_any = any(p in expected_pages for p in selected_pages) if expected_pages else None

            rows.append(
                {
                    "id": item["id"],
                    "mode": mode,
                    "question_type": item["question_type"],
                    "expected_path": expected_path,
                    "actual_path": retrieval.answer_path,
                    "route_match": retrieval.answer_path == expected_path,
                    "expected_pages": sorted(expected_pages),
                    "selected_pages": selected_pages,
                    "top1_page_hit": page_hit_top1,
                    "any_page_hit": page_hit_any,
                    "selected_documents": [d["document_name"] for d in retrieval.selected_documents],
                    "route_reason": retrieval.route_reason,
                }
            )
            print(
                f"[{item['id']:8}/{mode:10}] expected={expected_path:6} actual={retrieval.answer_path:6} "
                f"top1_hit={page_hit_top1} pages={selected_pages} reason={retrieval.route_reason}"
            )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_{ts}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # vision 기대 15문항 기준 요약(사용자 관심사: rag2가 실제로 vision을 타는가)
    vision_items = [r for r in rows if r["expected_path"] == "vision"]
    vision_hits_catalog = sum(1 for r in vision_items if r["mode"] == "catalog" and r["actual_path"] == "vision")
    vision_hits_nocat = sum(1 for r in vision_items if r["mode"] == "no_catalog" and r["actual_path"] == "vision")
    n_vision_q = len(vision_items) // 2  # catalog+no_catalog 두 번씩 있으므로

    print("\n=== 라우팅 사전점검 요약 ===")
    print(f"vision 기대 문항 수: {n_vision_q}")
    print(f"  catalog 모드에서 실제 vision 도달: {vision_hits_catalog}/{n_vision_q}")
    print(f"  no_catalog 모드에서 실제 vision 도달: {vision_hits_nocat}/{n_vision_q}")
    print(f"\n저장됨: {out_path}")


if __name__ == "__main__":
    main()

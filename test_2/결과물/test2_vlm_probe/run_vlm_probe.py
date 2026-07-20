"""test_2(rag2) VLM/vision 경로 검증 실험: 20문항 x {catalog, no_catalog} = 40회.

test12_total_test/run_final_experiment.py의 run_test2() 패턴을 그대로 따른다:
- run_single_test2.py를 수정 없이 독립 서브프로세스로 매 (mode, question)마다 새로 띄운다.
- test_2/rag2/index·cache는 기존 ingest 결과를 읽기전용으로 재사용(재-ingest 없음).
- 산출물은 이 폴더(results/)에만 쓴다. test_2/ 아래에는 어떤 파일도 새로 생기지 않는다
  (run_single_test2.py 자체의 계약과 동일).

calibrate_routing.py로 사전점검한 라우팅 기대치(expected_answer_path)를 각 결과에
route_match로 덧붙여, "실제로 vision을 밟았는가"를 이 스크립트 결과만으로 바로 집계할 수 있게 한다.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN_SINGLE_TEST2 = HERE.parent / "run_single_test2.py"
DATASET = HERE / "vlm_probe_dataset.json"
RESULTS_DIR = HERE / "results"

MODES = ("catalog", "no_catalog")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def warmup_models() -> None:
    log("모델 워밍업 중 (LLM/VLM/Embedding, 2개 모드 공정성을 위해 1회만)...")
    import ollama

    t0 = time.monotonic()
    ollama.chat(model="gemma4:12b", messages=[{"role": "user", "content": "안녕"}], keep_alive="10m")
    ollama.embed(model="embeddinggemma", input="warmup")
    log(f"워밍업 완료 ({time.monotonic() - t0:.1f}s)")


def run_subprocess(args: list[str], timeout: int) -> tuple[subprocess.CompletedProcess, float]:
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    wall = time.monotonic() - t0
    return proc, wall


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    items = json.loads(DATASET.read_text(encoding="utf-8"))
    log(f"질문 {len(items)}개 x 모드 2개(catalog/no_catalog) = {len(items) * 2}회 실행 시작")

    warmup_models()

    all_results: list[dict] = []
    for mode in MODES:
        out_dir = RESULTS_DIR / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            qid = item["id"]
            log(f"=== [{mode}] {qid} 실행 중 (기존 인덱스/캐시 읽기전용 재사용)...")

            result_path = out_dir / f"{qid}_result.json"
            args = [
                str(RUN_SINGLE_TEST2),
                "--mode", mode,
                "--qid", qid,
                "--question", item["question"],
                "--expected-documents", json.dumps(item["expected_documents"], ensure_ascii=False),
                "--expected-pages", json.dumps(item["expected_pages"], ensure_ascii=False),
                "--expected-keywords", json.dumps(item["expected_answer_keywords"], ensure_ascii=False),
                "--result-out", str(result_path),
            ]
            try:
                proc, wall = run_subprocess(args, timeout=1200)
            except subprocess.TimeoutExpired:
                log("    !! 타임아웃 (1200s 초과)")
                all_results.append(
                    {"pipeline": "test2", "mode": mode, "qid": qid, "error": "subprocess timeout (1200s)"}
                )
                continue

            if proc.returncode != 0:
                log(f"    !! 실패 (exit={proc.returncode}): {proc.stderr[-500:]}")
            else:
                log(f"    완료: {proc.stdout.strip()} (프로세스 전체 wall={wall:.1f}s)")

            if not result_path.exists():
                all_results.append(
                    {
                        "pipeline": "test2",
                        "mode": mode,
                        "qid": qid,
                        "error": f"결과 파일 미생성 (exit={proc.returncode})",
                        "subprocess_stderr_tail": proc.stderr[-4000:],
                    }
                )
                continue

            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["process_wall_seconds"] = round(wall, 3)
            result["subprocess_returncode"] = proc.returncode
            if proc.returncode != 0:
                result["subprocess_stderr_tail"] = proc.stderr[-4000:]

            expected_path = item["expected_answer_path"]
            result["expected_answer_path"] = expected_path
            result["route_match"] = result.get("answer_path") == expected_path
            result["question_type"] = item["question_type"]
            result["category"] = item["category"]

            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            all_results.append(result)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    combined_path = RESULTS_DIR / f"all_results_{ts}.json"
    combined_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"전체 결과 저장: {combined_path}")

    n_error = sum(1 for r in all_results if r.get("error"))
    n_vision_calls = sum(1 for r in all_results if r.get("model_calls", {}).get("vision_answer_calls", 0) > 0)
    log(f"완료: 총 {len(all_results)}건, 실패 {n_error}건, vision_answer_calls>=1인 실행 {n_vision_calls}건")


if __name__ == "__main__":
    main()

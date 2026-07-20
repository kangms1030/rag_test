"""4개 비교군(test1.catalog/no_catalog, test2.catalog/no_catalog) x 3질문 오케스트레이터.

- test_1: 매 (mode, question) 실행 직전 _cold_baseline_test1/{index,cache,output}
  (VLM 미호출 post-ingest 스냅샷)을 _work_test1/에 복원한 뒤 독립 서브프로세스로 실행한다.
  이전 질문이 만든 요약/청크 캐시가 다음 질문에 넘어가지 않도록 매번 냉시작을 보장한다.
- test_2: ingest 때 969페이지를 이미 전량 사전 색인해 두었고 ask 경로는 인덱스/캐시에
  쓰지 않으므로(test_2_timecost/results/REPORT.md에서 실측 확인) 스냅샷 복원이 필요 없다.
  기존 test_2/rag2/index·cache를 읽기전용으로 그대로 재사용한다.
- 두 파이프라인 모두 OS 프로세스 경계로 실행을 격리한다(모듈 전역 캐시/contextvar 등
  프로세스 내부 상태가 실행 간 새는 것을 원천 차단).
- 모든 비교군이 같은 Ollama 서버의 같은 모델(gemma4:12b, embeddinggemma)을 쓰므로, 첫
  비교군만 모델 로드 비용을 떠안지 않도록 실험 시작 시 1회 워밍업한다.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_TEST1 = HERE / "_cold_baseline_test1"
WORK_TEST1 = HERE / "_work_test1"
QA_FILE = HERE / "final_qa_dataset.json"

TEST1_MODES = ("catalog", "no_catalog")
TEST2_MODES = ("catalog", "no_catalog")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def warmup_models() -> None:
    log("모델 워밍업 중 (LLM/VLM/Embedding 서버 로드, 4개 비교군 공정성을 위해 실험 시작 시 1회만)...")
    import ollama

    t0 = time.monotonic()
    ollama.chat(model="gemma4:12b", messages=[{"role": "user", "content": "안녕"}], keep_alive="10m")
    ollama.embed(model="embeddinggemma", input="warmup")
    log(f"워밍업 완료 ({time.monotonic() - t0:.1f}s)")


def restore_test1_snapshot() -> None:
    for name in ("index", "cache", "output"):
        dst = WORK_TEST1 / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(BASELINE_TEST1 / name, dst)


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


def _attach_process_meta(result_path: Path, proc: subprocess.CompletedProcess, wall: float) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["process_wall_seconds"] = round(wall, 3)
    result["subprocess_returncode"] = proc.returncode
    if proc.returncode != 0:
        result["subprocess_stderr_tail"] = proc.stderr[-4000:]
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_test1(qa_items: list[dict], all_results: list[dict]) -> None:
    for mode in TEST1_MODES:
        out_dir = HERE / f"test1_{mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in qa_items:
            qid = item["id"]
            log(f"=== [test1/{mode}] {qid} 냉시작 스냅샷 복원 중...")
            restore_test1_snapshot()

            result_path = out_dir / f"{qid}_result.json"
            args = [
                str(HERE / "run_single_test1.py"),
                "--mode", mode,
                "--qid", qid,
                "--question", item["question"],
                "--index-dir", str(WORK_TEST1 / "index"),
                "--cache-dir", str(WORK_TEST1 / "cache"),
                "--output-dir", str(WORK_TEST1 / "output"),
                "--expected-documents", json.dumps(item["expected_documents"], ensure_ascii=False),
                "--expected-pages", json.dumps(item["expected_pages"], ensure_ascii=False),
                "--expected-keywords", json.dumps(item["expected_answer_keywords"], ensure_ascii=False),
                "--result-out", str(result_path),
            ]
            log(f"    실행 중 (냉시작, 상위 12페이지 VLM 요약 포함 가능 — 수 분 소요될 수 있음)...")
            try:
                proc, wall = run_subprocess(args, timeout=2400)
            except subprocess.TimeoutExpired:
                log(f"    !! 타임아웃 (2400s 초과)")
                all_results.append({"pipeline": "test1", "mode": mode, "qid": qid, "error": "subprocess timeout (2400s)"})
                continue

            if proc.returncode != 0:
                log(f"    !! 실패 (exit={proc.returncode}): {proc.stderr[-500:]}")
            else:
                log(f"    완료: {proc.stdout.strip()} (프로세스 전체 wall={wall:.1f}s)")

            if not result_path.exists():
                all_results.append(
                    {
                        "pipeline": "test1",
                        "mode": mode,
                        "qid": qid,
                        "error": f"결과 파일 미생성 (exit={proc.returncode})",
                        "subprocess_stderr_tail": proc.stderr[-4000:],
                    }
                )
                continue

            result = _attach_process_meta(result_path, proc, wall)

            # _work_test1은 다음 질문에서 스냅샷으로 덮어써지므로, evidence 이미지를 비교군
            # 결과 폴더로 먼저 복사해 보존한다(요구사항 5: 질문별 독립 폴더에 결과 저장).
            run_id = result.get("run_id")
            if run_id:
                src_ev = WORK_TEST1 / "output" / "evidence" / run_id
                if src_ev.exists():
                    dst_ev = out_dir / f"{qid}_evidence"
                    if dst_ev.exists():
                        shutil.rmtree(dst_ev)
                    shutil.copytree(src_ev, dst_ev)
                    result["evidence_dir"] = str(dst_ev)
                    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            all_results.append(result)


def run_test2(qa_items: list[dict], all_results: list[dict]) -> None:
    for mode in TEST2_MODES:
        out_dir = HERE / f"test2_{mode}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in qa_items:
            qid = item["id"]
            log(f"=== [test2/{mode}] {qid} 실행 중 (기존 인덱스/캐시 읽기전용 재사용)...")

            result_path = out_dir / f"{qid}_result.json"
            args = [
                str(HERE / "run_single_test2.py"),
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
                log(f"    !! 타임아웃 (1200s 초과)")
                all_results.append({"pipeline": "test2", "mode": mode, "qid": qid, "error": "subprocess timeout (1200s)"})
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

            result = _attach_process_meta(result_path, proc, wall)
            all_results.append(result)


def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    qa_items = json.loads(QA_FILE.read_text(encoding="utf-8"))
    log(f"질문 {len(qa_items)}개 x 비교군 4개(test1.catalog/no_catalog, test2.catalog/no_catalog) = {len(qa_items) * 4}회 실행 시작")

    warmup_models()

    all_results: list[dict] = []
    run_test1(qa_items, all_results)
    run_test2(qa_items, all_results)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    combined_path = HERE / f"all_results_{ts}.json"
    combined_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"전체 결과 저장: {combined_path}")

    n_error = sum(1 for r in all_results if r.get("error"))
    log(f"완료: 총 {len(all_results)}건, 실패 {n_error}건")


if __name__ == "__main__":
    main()

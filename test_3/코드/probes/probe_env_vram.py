"""P0-E: VRAM 상주 크기 + 12b↔e4b 스왑 지연 실측.

목적(계획 F5 검증): gemma4:12b(7.6GB) + gemma4:e4b(9.6GB) 합 17.2GB > 16GB 이므로
동시 상주가 불가능하고, e4b를 온라인 중간 단계에 쓰면 매 호출 모델 스왑이 발생한다는 가설을
`ollama ps`의 실측 VRAM과 chat 응답의 load_duration으로 확인한다.

모델 호출은 짧은 프롬프트 1개씩(각 모델 warmup)뿐 — 답변 품질은 보지 않는다.
결과: test_3/probes/results/p0e_vram.json
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

import ollama

RESULTS = Path(__file__).resolve().parent / "results" / "p0e_vram.json"
NUM_CTX_MAIN = 16384   # 계획상 12b 온라인 num_ctx
NUM_CTX_LIGHT = 2048   # 계획상 e4b 판정용 축소 num_ctx


def ollama_ps() -> list[dict]:
    """`ollama ps` 파싱 → [{name, size, processor, until}]. SIZE는 실제 VRAM+RAM 점유."""
    out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, encoding="utf-8").stdout
    lines = [l for l in out.splitlines() if l.strip()]
    rows = []
    for l in lines[1:]:  # skip header
        # 컬럼: NAME  ID  SIZE(2토큰: "7.6 GB")  PROCESSOR(가변)  UNTIL(가변)
        parts = l.split()
        if len(parts) < 4:
            continue
        name = parts[0]
        # SIZE = 숫자+단위 찾기
        size = "?"
        proc = "?"
        for i in range(len(parts) - 1):
            if parts[i].replace(".", "").isdigit() and parts[i + 1] in ("GB", "MB"):
                size = f"{parts[i]} {parts[i+1]}"
                proc = " ".join(parts[i + 2:i + 4]) if i + 2 < len(parts) else "?"
                break
        rows.append({"name": name, "size": size, "processor": proc, "raw": l.strip()})
    return rows


def warm(model: str, num_ctx: int) -> dict:
    """짧은 호출로 모델 로드 후 타이밍 반환. load_duration이 곧 (재)로드/스왑 지연."""
    t0 = time.time()
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": "안녕. 한 단어로만 답해."}],
        options={"num_ctx": num_ctx, "temperature": 0.0},
        keep_alive="10m",
    )
    wall = time.time() - t0
    return {
        "model": model,
        "num_ctx": num_ctx,
        "wall_s": round(wall, 2),
        "load_duration_s": round(resp.get("load_duration", 0) / 1e9, 2),
        "total_duration_s": round(resp.get("total_duration", 0) / 1e9, 2),
        "eval_count": resp.get("eval_count"),
    }


def unload(model: str) -> None:
    """keep_alive=0으로 즉시 언로드."""
    try:
        ollama.chat(model=model, messages=[{"role": "user", "content": "x"}],
                    options={"num_ctx": 512}, keep_alive=0)
    except Exception:
        pass


def main() -> None:
    log: dict = {"steps": []}

    def snap(label: str):
        ps = ollama_ps()
        entry = {"label": label, "ps": ps}
        log["steps"].append(entry)
        print(f"\n[{label}] ollama ps:")
        for r in ps:
            print(f"    {r['name']:26s} {r['size']:>8s}  {r['processor']}")
        if not ps:
            print("    (상주 모델 없음)")
        return entry

    # 0) 시작 상태 정리
    print("=== P0-E VRAM/스왑 실측 시작 ===")
    for m in ("gemma4:12b", "gemma4:e4b", "embeddinggemma:latest"):
        unload(m)
    time.sleep(2)
    snap("baseline(전부 언로드)")

    # 1) embeddinggemma 단독
    log.setdefault("warm", {})["embed"] = None
    try:
        r = ollama.embed(model="embeddinggemma", input="테스트 문장")
        log["warm"]["embed"] = "ok"
    except Exception as e:
        log["warm"]["embed"] = f"err: {e}"
    snap("embeddinggemma 상주")

    # 2) 12b 단독 (num_ctx 16384)
    log["warm"]["12b_ctx16k"] = warm("gemma4:12b", NUM_CTX_MAIN)
    print(f"    12b warm: {log['warm']['12b_ctx16k']}")
    snap("12b(16k) + embeddinggemma 동시")

    # 3) e4b 로드 (num_ctx 2048) — 12b가 유지되는지 스왑되는지 관찰
    log["warm"]["e4b_ctx2k_first"] = warm("gemma4:e4b", NUM_CTX_LIGHT)
    print(f"    e4b warm(1st): {log['warm']['e4b_ctx2k_first']}")
    snap("e4b(2k) 로드 직후 - 12b 상주 여부 관찰")

    # 4) 다시 12b 호출 — 스왑되어 재로드되는지(load_duration 큼) 측정
    log["warm"]["12b_after_e4b"] = warm("gemma4:12b", NUM_CTX_MAIN)
    print(f"    12b warm(after e4b): {log['warm']['12b_after_e4b']}")
    snap("12b 재호출 후")

    # 5) 다시 e4b — 스왑 지연 재측정
    log["warm"]["e4b_after_12b"] = warm("gemma4:e4b", NUM_CTX_LIGHT)
    print(f"    e4b warm(after 12b): {log['warm']['e4b_after_12b']}")
    snap("e4b 재호출 후")

    # 판정
    e4b_reload = log["warm"]["e4b_after_12b"]["load_duration_s"]
    b12_reload = log["warm"]["12b_after_e4b"]["load_duration_s"]
    log["verdict"] = {
        "co_resident_12b_e4b": None,  # 아래 채움
        "swap_load_12b_s": b12_reload,
        "swap_load_e4b_s": e4b_reload,
        "note": "load_duration_s가 크면(>3s) 재로드=스왑 발생. 0에 가까우면 상주 유지.",
    }
    # e4b 로드 직후 스냅샷에서 12b가 사라졌으면 동시상주 불가
    after_e4b_ps = log["steps"][3]["ps"]
    names = {r["name"] for r in after_e4b_ps}
    has_12b = any("gemma4:12b" in n for n in names)
    has_e4b = any("gemma4:e4b" in n for n in names)
    log["verdict"]["co_resident_12b_e4b"] = bool(has_12b and has_e4b)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 판정 ===")
    print(f"  12b+e4b 동시상주: {log['verdict']['co_resident_12b_e4b']}")
    print(f"  12b 재로드(스왑) 지연: {b12_reload}s")
    print(f"  e4b 재로드(스왑) 지연: {e4b_reload}s")
    print(f"\n결과 저장: {RESULTS}")


if __name__ == "__main__":
    main()

"""[임시/독립 · 프로젝트 최상위] 터미널에서 직접 질문하며 답변 LLM을 바꿔 비교하는 대화형 CLI.

test_3/코드/ask_cli.py 와 사용감이 같다(같은 전체 파이프라인: 검색→답변→검증→롤백, 같은 출력 형식).
차이는 단 하나 — 답변 LLM(text/vision)을 시작 시 --model 로 고르고, 세션 중 `:model <이름>`으로
즉시 바꿔 '같은 질문'을 두 모델로 물어 결과/소요시간을 바로 비교할 수 있다는 것.

설계 원칙(사용자 요청):
  - 기존 코드 무수정. rag3는 import만 하고, 모델 치환은 load_config의 기존 overrides 인자로만 한다
    (원본 config.yaml 불변). verify_model은 config상 ""라 text_answer_model을 따라가므로 자동 치환.
  - 임베딩(embeddinggemma)·리랭커(bge)는 그대로 두고 답변 LLM만 바꾼다(=단순 치환).
  - 프로젝트 최상위 단일 파일 → 실험 끝나면 이 파일 하나만 지우면 정리 끝.

사용법 (conda 환경 intern_chatbot):
  # e4b로 대화형 시작(기본). 질문을 계속 입력
  cmd /c conda activate intern_chatbot && python ask_compare_cli.py

  # 12b로 시작하려면
  python ask_compare_cli.py --model gemma4:12b

  # 세션 안에서 모델 전환(같은 질문을 두 모델로 비교):
  질문> :model gemma4:12b      (또는  :model gemma4:e4b)
  질문> :model                 현재 모델 확인

  # 한글 입력이 콘솔에서 깨질 때: UTF-8 텍스트 파일로 질문 전달
  python ask_compare_cli.py --file 질문.txt

종료: exit / quit / q / 빈 줄, 또는 Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --- rag3 패키지 import 경로(최상위에서 test_3/코드 로) ---
_HERE = Path(__file__).resolve().parent          # 프로젝트 최상위
_CODE = _HERE / "test_3" / "코드"
sys.path.insert(0, str(_CODE))

# --- Windows 한글 콘솔 인코딩(입출력 UTF-8) ---
try:
    if os.name == "nt":
        os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag3.config import load_config             # noqa: E402
from rag3.controller import answer_question      # noqa: E402
from rag3.evidence import export_evidence, resolve_evidence  # noqa: E402
from rag3.models import get_backend              # noqa: E402
from rag3.utils import new_run_id                # noqa: E402


def _print_trace(r: dict, config) -> None:
    """질문 1건의 단계/모델/호출수를 rag3 metrics로부터 복원해 출력(rag3 무수정)."""
    m = r.get("metrics", {})
    t = m.get("timings_seconds", {}) or {}
    llm = config.text_answer_model
    vmodel = config.verify_model or config.text_answer_model
    emb = config.embedding_model
    rr = config.rerank_model.split("/")[-1]

    embed_n = m.get("embed_calls", 0)
    rerank_n = m.get("rerank_calls", 0)
    text_n = m.get("text_answer_calls", 0)
    vision_n = m.get("vision_answer_calls", 0)
    judge_n = m.get("judge_calls", 0)
    verify_n = m.get("verify_calls", 0)
    lretry = m.get("length_retry_count", 0)
    rollback_actions = [h.get("action") for h in r.get("rollback_history", [])]

    rows: list[tuple[str, str, int, str]] = []
    rows.append(("S1 질문 임베딩", emb, embed_n, "질문→벡터" + ("(재검색 포함)" if embed_n > 1 else "")))
    rows.append(("S2 하이브리드 검색", "BM25+dense (모델 아님)", 0, "청크 top20 후보"))
    rows.append(("S3 리랭크", rr, rerank_n, "후보 재정렬→상위 페이지 승격"))
    if judge_n:
        rows.append(("S3a CRAG 질의재작성", llm, judge_n, "검색 실패→질문 재작성 후 재검색"))
    if text_n:
        note = f"text 답변 생성{' (굶음 재생성 포함)' if text_n > 1 else ''}"
        if lretry:
            note += f" +내부 length재발행 {lretry}회"
        rows.append(("S6 답변(text)", llm, text_n, note))
    if vision_n:
        rows.append(("S6 답변(vision)", config.vision_answer_model, vision_n, "이미지 전사-후-답변"))
    if verify_n:
        rows.append(("S7 검증(groundedness)", vmodel, verify_n, "근거 뒷받침 여부 판정"))
    if rollback_actions:
        rows.append(("S8 롤백", "결정론 (모델 아님)", 0, ", ".join(a for a in rollback_actions if a)))

    print("[파이프라인 추적]")
    for label, model, n, note in rows:
        cnt = f"×{n}" if n else "  "
        print(f"  {label:<18} {cnt:<4} {model:<22} {note}")
    llm_total = text_n + vision_n + judge_n + verify_n
    print(f"  ── 합계: LLM({llm}) {llm_total}회"
          + (f"(+length재발행 {lretry})" if lretry else "")
          + f" · 리랭커 {rerank_n}회 · 임베딩 {embed_n}회")
    if t:
        seg = "  ".join(f"{k}={v}s" for k, v in t.items())
        print(f"  ── 타이밍: {seg}")


def _print_result(question: str, r: dict, dt: float, config, save_evidence: bool = False) -> None:
    v = r.get("verification") or {}
    print("\n" + "=" * 68)
    print(f"[질문] {question}")
    print(f"[모델] {config.text_answer_model}   [경로] {r['answer_path']}   [신뢰도] {r.get('confidence')}")
    print("-" * 68)
    print(r.get("final_answer", "").strip() or "(빈 응답)")
    print("-" * 68)
    pages = r.get("selected_pages", [])
    if pages:
        print("[근거 페이지]")
        evidence = resolve_evidence(r, config)
        for p, e in zip(pages, evidence):
            print(f"  · {p.get('document_name')}  p{p.get('page_number')}  (rerank={p.get('page_score')})")
            if e.get("page_image_resolved"):
                print(f"    이미지: {e['page_image_resolved']}")
    if v:
        flags = []
        if v.get("unsupported_claims"):
            flags.append(f"미지원숫자={v['unsupported_claims']}")
        if v.get("transcription_ocr_mismatch"):
            flags.append("전사-OCR불일치")
        if v.get("abstain"):
            flags.append("회피")
        print(f"[검증] {'· '.join(flags) if flags else 'OK'}")
    print("-" * 68)
    _print_trace(r, config)
    if save_evidence:
        files = export_evidence(r, config, r.get("run_id") or new_run_id(question))
        if files:
            print(f"[근거 이미지 저장] {Path(files[0]['file']).parent}")
    print(f"[소요] {dt:.1f}s")
    print("=" * 68)


def _ask_once(question: str, config, backend, save_evidence: bool = False) -> None:
    question = (question or "").strip()
    if not question:
        return
    print(f"\n... 답변 생성 중 [{config.text_answer_model}] (첫 질문/모델 전환 직후는 로딩으로 느릴 수 있음)")
    t0 = time.time()
    r = answer_question(question, config, backend)
    _print_result(question, r, time.time() - t0, config, save_evidence=save_evidence)


def _ollama_bin() -> str:
    """ollama 실행파일 경로(PATH 우선, 없으면 기본 설치 경로)."""
    return shutil.which("ollama") or str(
        Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe")


def _unload_model(model_name: str) -> None:
    """모델을 VRAM에서 즉시 내림(`ollama stop`). 두 모델이 동시에 상주해 전용 VRAM을 넘겨
    공유 메모리(느림)로 흘러넘치는 것을 막는다 — 모델 전환 시 이전 모델을 반드시 내린다."""
    if not model_name:
        return
    try:
        subprocess.run([_ollama_bin(), "stop", model_name],
                       check=False, capture_output=True, timeout=30)
        print(f"[언로드] {model_name} → VRAM에서 내림")
    except Exception as e:
        print(f"[언로드 경고] {model_name}: {e!r}")


def _warm_model(backend, config) -> None:
    """현재 config.text_answer_model 을 VRAM에 상주시켜(더미 1토큰) 비교 시 콜드 편향 제거."""
    try:
        print(f"[워밍업] {config.text_answer_model} 상주...", flush=True)
        backend.chat_text("답변은 '준비'라고만 하세요.")
    except Exception as e:
        print(f"[워밍업 경고] {e!r}")


def _switch_model(arg: str, config, backend) -> None:
    """세션 중 답변 LLM 전환. config만 in-memory로 바꾸고(rag3/원본 config.yaml 불변),
    이전 모델은 VRAM에서 내려 '한 번에 한 모델만' 상주하게 한다(공유 메모리 넘침 방지)."""
    name = arg.strip()
    if not name:
        print(f"[현재 모델] text/vision = {config.text_answer_model}"
              f"  (verify = {config.verify_model or config.text_answer_model})")
        return
    old = config.text_answer_model
    config.text_answer_model = name
    config.vision_answer_model = name
    # verify_model이 ""면 자동으로 text를 따라감. 비어있지 않게 명시돼 있었다면 그대로 둔다.
    if old and old != name:
        _unload_model(old)  # 새 모델을 올리기 전에 이전 모델을 내려 2개 겹침 방지
    print(f"[모델 전환] → {name}")
    _warm_model(backend, config)


def _repl(config, backend, save_evidence: bool = False) -> None:
    print("\n대화형 비교 모드. 질문을 입력하세요.")
    print("  · 모델 전환(이전 모델 자동 언로드):  :model gemma4:12b  /  :model gemma4:e4b  /  :model (현재확인)")
    print("  · 현재 모델 수동 언로드(VRAM 비우기):  :unload")
    print("  · 종료:  exit / quit / q / 빈 줄, 또는 Ctrl+C")
    while True:
        try:
            q = input(f"\n[{config.text_answer_model}] 질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            _unload_model(config.text_answer_model)  # 나갈 때 VRAM 정리
            return
        if not q or q.lower() in {"exit", "quit", "q"}:
            print("종료합니다.")
            _unload_model(config.text_answer_model)  # 나갈 때 VRAM 정리
            return
        if q.startswith(":model"):
            _switch_model(q[len(":model"):], config, backend)
            continue
        if q.lower() == ":unload":
            _unload_model(config.text_answer_model)
            continue
        try:
            _ask_once(q, config, backend, save_evidence=save_evidence)
        except Exception as e:  # 한 질문 실패가 세션을 죽이지 않게
            print(f"[오류] 답변 생성 실패: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="rag3 파이프라인으로 답변 LLM을 바꿔가며 터미널에서 직접 비교")
    ap.add_argument("--model", default="gemma4:e4b",
                    help="답변 LLM(text/vision) 초기값. 기본 gemma4:e4b. 예: --model gemma4:12b")
    ap.add_argument("-q", "--question", default=None, help="질문 문자열(한글이 깨지면 --file 사용)")
    ap.add_argument("--file", default=None, help="UTF-8로 저장된 질문 텍스트 파일 경로")
    ap.add_argument("--config", default=None, help="config.yaml 경로(기본: rag3/config.yaml)")
    ap.add_argument("--save-evidence", action="store_true", help="근거 페이지 이미지를 저장")
    ap.add_argument("--no-warm", action="store_true", help="시작 시 모델 워밍업 생략")
    args = ap.parse_args()

    print(f"설정/모델/인덱스 로딩 중... (답변 LLM = {args.model})")
    # 기존 코드 무수정: load_config의 overrides로 답변 모델만 치환(원본 config.yaml 불변)
    # keep_alive를 5m로 낮춰 모델이 오래 상주하다 고아 러너로 남는 위험을 줄인다(비교용 세션 한정).
    config = load_config(args.config, overrides={
        "text_answer_model": args.model,
        "vision_answer_model": args.model,
        "ollama_keep_alive": "5m",
    })
    config.ensure_dirs()
    backend = get_backend(config)

    try:
        from rag3.rerank import get_reranker
        print("리랭커 로딩 중...")
        get_reranker(config)
    except ImportError as e:
        print(f"\n[환경 오류] 필수 패키지를 찾을 수 없습니다: {e}")
        print("  -> intern_chatbot conda 환경에서 실행해야 합니다:")
        print("     cmd /c conda activate intern_chatbot && python ask_compare_cli.py")
        print(f"  (현재 파이썬: {sys.executable})")
        raise SystemExit(1)

    if not args.no_warm:
        _warm_model(backend, config)
    print(f"준비 완료 (source_parsed={config.source_parsed.name}, "
          f"retry_on_length={config.ollama_retry_on_length})")

    if args.file:
        _ask_once(Path(args.file).read_text(encoding="utf-8"), config, backend,
                  save_evidence=args.save_evidence)
    elif args.question:
        _ask_once(args.question, config, backend, save_evidence=args.save_evidence)
    else:
        _repl(config, backend, save_evidence=args.save_evidence)


if __name__ == "__main__":
    main()

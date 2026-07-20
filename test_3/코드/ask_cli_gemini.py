"""Gemini(Flash-lite) 백엔드로 돌리는 간단 확인용 질의 CLI (외부 API RAG 대조군 시연).

ask_cli.py의 Gemini 버전. 검색·리랭커·임베딩은 로컬 그대로 두고, 답변·검증만
gemini-3.1-flash-lite로 처리한다(rag3x.Rag3xEngine, x_backend=gemini). 기존 rag3 코드는
수정하지 않는다(import 전용). 로컬판과 나란히 돌려 "외부 API RAG가 이만큼 빠르다"를 확인하는 용도다.

준비:
  - intern_chatbot conda 환경
  - 프로젝트 루트 .env에 GEMINI_API_KEY (키는 화면/로그에 출력하지 않음)
  - API 장애 시 자동으로 로컬 12b로 폴백

사용법:
  python ask_cli_gemini.py                       # 대화형(권장)
  python ask_cli_gemini.py -q "제주교육청 스쿨넷 5단계 요금은?"
  python ask_cli_gemini.py --file 질문.txt        # 한글이 콘솔에서 깨질 때
종료: exit / quit / q / 빈 줄, 또는 Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent  # .../test_3
sys.path.insert(0, str(_HERE))

# Windows 한글 콘솔 UTF-8
try:
    if os.name == "nt":
        os.system("chcp 65001 > nul")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass


def _print_result(question: str, r: dict, dt: float) -> None:
    m = r.get("metrics", {})
    print("\n" + "=" * 60)
    print(f"[질문] {question}")
    print(f"[경로] {r.get('answer_path')}   [신뢰도] {r.get('confidence')}")
    print("-" * 60)
    print((r.get("final_answer") or "").strip() or "(빈 응답)")
    print("-" * 60)
    for p in r.get("selected_pages", []):
        print(f"  · {p.get('document_name')}  p{p.get('page_number')}  (rerank={p.get('page_score')})")
    # Gemini 계측(있으면): 호출수 / 토큰 / 비용 / 순수 API 지연
    if m.get("gemini_calls") is not None:
        print(f"[Gemini] 호출 {m.get('gemini_calls')}회 · "
              f"토큰 in {m.get('gemini_tokens_in')}/out {m.get('gemini_tokens_out')} · "
              f"API {m.get('gemini_api_s')}s · 비용≈${m.get('gemini_cost')}")
    else:
        print("[Gemini] (호출 없음 — 무관질문 거절 또는 로컬 폴백)")
    print(f"[소요] {dt:.1f}s   [모델] {r.get('metrics', {}).get('total_model_calls', '?')} 모델호출")
    print("=" * 60)


def _ask_once(engine, question: str) -> None:
    question = (question or "").strip()
    if not question:
        return
    print("\n... 답변 생성 중 (첫 질문은 로딩으로 느릴 수 있음)")
    t0 = time.time()
    r = engine.ask(question)
    _print_result(question, r, time.time() - t0)


def _repl(engine) -> None:
    print("\n대화형 질의 모드 (Gemini Flash-lite). 질문 입력. (종료: exit/quit/q/빈 줄, Ctrl+C)")
    while True:
        try:
            q = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return
        if not q or q.lower() in {"exit", "quit", "q"}:
            print("종료합니다.")
            return
        try:
            _ask_once(engine, q)
        except Exception as e:
            print(f"[오류] 답변 생성 실패: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini Flash-lite 백엔드로 rag3 파이프라인 질의(확인용)")
    ap.add_argument("-q", "--question", default=None, help="질문 문자열(한글 깨지면 --file)")
    ap.add_argument("--file", default=None, help="UTF-8 질문 텍스트 파일 경로")
    ap.add_argument("--model", default=None, help="Gemini 모델명 override(기본 gemini-3.1-flash-lite)")
    args = ap.parse_args()

    print("설정/모델/인덱스 로딩 중 (검색·리랭커는 로컬, 답변·검증은 Gemini)...")
    x_over = {"x_backend": "gemini"}
    if args.model:
        x_over["x_gemini_model"] = args.model
    try:
        from rag3x import Rag3xEngine
        engine = Rag3xEngine(x_overrides=x_over)
    except ImportError as e:
        print(f"\n[환경 오류] 필수 패키지 없음: {e}")
        print("  -> intern_chatbot conda 환경에서 실행하세요.")
        print(f"  (현재 파이썬: {sys.executable})")
        raise SystemExit(1)
    except RuntimeError as e:
        print(f"\n[키 오류] {e}")
        print("  -> 프로젝트 루트 .env에 GEMINI_API_KEY를 설정하세요.")
        raise SystemExit(1)

    print(f"준비 완료 (backend=gemini, model={engine.config.x_gemini_model})")

    if args.file:
        _ask_once(engine, Path(args.file).read_text(encoding="utf-8"))
    elif args.question:
        _ask_once(engine, args.question)
    else:
        _repl(engine)


if __name__ == "__main__":
    main()

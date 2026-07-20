"""독립 실행 질의 CLI — 완성된 rag3 파이프라인을 import만 해서 터미널에서 직접 질문한다.

기존 rag3/ 코드는 전혀 수정하지 않는다(import 전용). 완성 파이프라인
`controller.answer_question`(검색 -> 답변 -> 검증 -> 결정론 롤백 + 콜드스타트 whitespace
런어웨이 방어)을 사용한다. 모델/인덱스는 시작 시 1회 로드하고 이후 대화형 루프에서 재사용하므로,
두 번째 질문부터는 모델이 웜 상태라 더 빠르고 콜드 폭주도 거의 없다.

사용법 (conda 환경 intern_chatbot):
  # 대화형(권장) — 질문을 계속 입력
  cmd /c conda activate intern_chatbot && cd test_3 && python ask_cli.py

  # 한 번만 물어보기(argv). 한글이 깨지면 --file 사용
  python ask_cli.py -q "제주교육청 스쿨넷 5단계 요금은?"

  # 한글 입력이 콘솔에서 깨질 때: UTF-8 텍스트 파일로 질문 전달
  python ask_cli.py --file 질문.txt

종료: 대화형에서 exit / quit / q / 빈 줄 입력, 또는 Ctrl+C.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# --- rag3 패키지 import 경로(어디서 실행하든 동작) ---
_HERE = Path(__file__).resolve().parent  # .../test_3
sys.path.insert(0, str(_HERE))

# --- Windows 한글 콘솔 인코딩(입출력 UTF-8) ---
try:
    if os.name == "nt":
        os.system("chcp 65001 > nul")  # 콘솔 코드페이지를 UTF-8로
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag3.config import load_config          # noqa: E402
from rag3.controller import answer_question   # noqa: E402
from rag3.evidence import export_evidence, resolve_evidence  # noqa: E402
from rag3.models import get_backend           # noqa: E402
from rag3.utils import new_run_id             # noqa: E402


def _print_trace(r: dict, config) -> None:
    """질문 1건이 어떤 모델을 몇 번, 어떤 단계로 거쳐 답까지 왔는지 재구성해 출력.

    rag3 metrics(embed/rerank/text/vision/judge/verify/rollback/length_retry)와
    rollback_history/route_reason으로부터 파이프라인을 그대로 복원한다(rag3 무수정).
    """
    m = r.get("metrics", {})
    t = m.get("timings_seconds", {}) or {}
    path = r.get("answer_path")

    llm = config.text_answer_model           # 답변/재작성/검증 공용 gemma
    vmodel = config.verify_model or config.text_answer_model
    emb = config.embedding_model
    rr = config.rerank_model.split("/")[-1]   # 리랭커 이름만

    embed_n = m.get("embed_calls", 0)
    rerank_n = m.get("rerank_calls", 0)
    text_n = m.get("text_answer_calls", 0)
    vision_n = m.get("vision_answer_calls", 0)
    judge_n = m.get("judge_calls", 0)
    verify_n = m.get("verify_calls", 0)
    lretry = m.get("length_retry_count", 0)
    rollback_actions = [h.get("action") for h in r.get("rollback_history", [])]

    # (라벨, 모델, 호출수, 비고) — 호출수 0인 선택 단계는 건너뜀
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
        rows.append((f"S6 답변(text)", llm, text_n, note))
    if vision_n:
        rows.append((f"S6 답변(vision)", config.vision_answer_model, vision_n, "이미지 전사-후-답변"))
    if verify_n:
        rows.append(("S7 검증(groundedness)", vmodel, verify_n, "근거 뒷받침 여부 판정"))
    if rollback_actions:
        rows.append(("S8 롤백", "결정론 (모델 아님)", 0, ", ".join(a for a in rollback_actions if a)))

    print("[파이프라인 추적]")
    for label, model, n, note in rows:
        cnt = f"×{n}" if n else "  "
        print(f"  {label:<18} {cnt:<4} {model:<22} {note}")

    # 합계: 실제 LLM(gemma) 호출은 답변+재작성+검증 전부 (기존 total_model_calls는 embed+답변만 셌음)
    llm_total = text_n + vision_n + judge_n + verify_n
    print(f"  ── 합계: LLM({llm}) {llm_total}회"
          + (f"(+length재발행 {lretry})" if lretry else "")
          + f" · 리랭커 {rerank_n}회 · 임베딩 {embed_n}회")
    if t:
        seg = "  ".join(f"{k}={v}s" for k, v in t.items())
        print(f"  ── 타이밍: {seg}")


def _print_result(question: str, r: dict, dt: float, config, save_evidence: bool = False) -> None:
    m = r.get("metrics", {})
    v = r.get("verification") or {}
    print("\n" + "=" * 68)
    print(f"[질문] {question}")
    print(f"[경로] {r['answer_path']}   [신뢰도] {r.get('confidence')}")
    print("-" * 68)
    print(r.get("final_answer", "").strip() or "(빈 응답)")
    print("-" * 68)
    pages = r.get("selected_pages", [])
    if pages:
        print("[근거 페이지]")
        evidence = resolve_evidence(r, config)  # 페이지 이미지 절대경로 부착(evidence는 pages와 같은 순서)
        for p, e in zip(pages, evidence):
            print(f"  · {p.get('document_name')}  p{p.get('page_number')}"
                  f"  (rerank={p.get('page_score')})")
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
    print("\n... 답변 생성 중 (첫 질문은 모델 로딩으로 느릴 수 있음)")
    t0 = time.time()
    r = answer_question(question, config, backend)
    _print_result(question, r, time.time() - t0, config, save_evidence=save_evidence)


def _repl(config, backend, save_evidence: bool = False) -> None:
    print("\n대화형 질의 모드. 질문을 입력하세요. (종료: exit / quit / q / 빈 줄, 또는 Ctrl+C)")
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
            _ask_once(q, config, backend, save_evidence=save_evidence)
        except Exception as e:  # 한 질문 실패가 세션을 죽이지 않게
            print(f"[오류] 답변 생성 실패: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="rag3 완성 파이프라인으로 터미널에서 직접 질문")
    ap.add_argument("-q", "--question", default=None, help="질문 문자열(한글이 깨지면 --file 사용)")
    ap.add_argument("--file", default=None, help="UTF-8로 저장된 질문 텍스트 파일 경로")
    ap.add_argument("--config", default=None, help="config.yaml 경로(기본: rag3/config.yaml)")
    ap.add_argument("--save-evidence", action="store_true",
                    help="근거 페이지 이미지를 outputs/evidence/<run_id>/에 복사해 저장")
    args = ap.parse_args()

    print("설정/모델/인덱스 로딩 중...")
    config = load_config(args.config)
    config.ensure_dirs()
    backend = get_backend(config)

    # 환경 사전 점검 + 리랭커 프리로드: 리랭커(sentence_transformers/torch)는 원래 첫 질문에서
    # 지연 로드되어, 잘못된 환경이면 첫 질문에서야 'No module named sentence_transformers'가 난다.
    # 여기서 미리 로드해 (1) 오류를 프롬프트 전에 즉시 표면화하고 (2) 첫 질문도 빠르게 한다.
    try:
        from rag3.rerank import get_reranker
        print("리랭커 로딩 중...")
        get_reranker(config)
    except ImportError as e:
        print(f"\n[환경 오류] 필수 패키지를 찾을 수 없습니다: {e}")
        print("  -> intern_chatbot conda 환경에서 실행해야 합니다. 예:")
        print("     cmd /c conda activate intern_chatbot && cd test_3 && python ask_cli.py")
        print(f"  (현재 파이썬: {sys.executable})")
        raise SystemExit(1)

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

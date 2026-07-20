from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("rag2")

_DEFAULT_EVAL_FILE = Path(__file__).resolve().parent.parent / "rag_eval_dataset.json"


def cmd_check(args: argparse.Namespace) -> None:
    from .config import load_config
    from .envcheck import check_environment

    config = None
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.warning("config 로드 실패, ollama 서버/인덱스 점검은 생략: %s", e)

    result = check_environment(config, check_indexes=not args.skip_indexes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_required_ok"]:
        raise SystemExit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    from .config import load_config
    from .ingest import run_ingest
    from .models import get_backend

    config = load_config(args.config)
    config.ensure_dirs()
    backend = get_backend(config)

    summary = run_ingest(config, backend, force=args.force, limit_docs=args.limit_docs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_add(args: argparse.Namespace) -> None:
    from .add_doc import add_documents, remove_document
    from .config import load_config
    from .models import get_backend

    config = load_config(args.config)
    config.ensure_dirs()
    backend = get_backend(config)

    if args.remove:
        out = remove_document(config, backend, args.remove)
    elif args.pdf:
        out = add_documents(config, backend, args.pdf,
                            run_vlm=not args.skip_vlm, force_parse=args.force_parse)
    else:
        raise SystemExit("--pdf <파일명> 또는 --remove <파일명|slug>가 필요합니다")
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    from .answer import generate_answer
    from .config import load_config
    from .eval_sets import get_eval_item
    from .metrics import RunMetrics, record_timing, run_metrics
    from .models import get_backend
    from .retrieve import run_retrieval
    from .schema import build_final_json, save_final_json
    from .utils import new_run_id

    config = load_config(args.config)
    config.ensure_dirs()
    backend = get_backend(config)

    if args.qid:
        item = get_eval_item(args.eval_file or _DEFAULT_EVAL_FILE, args.qid)
        question = item["question"]
    elif args.question_file:
        question = Path(args.question_file).read_text(encoding="utf-8").strip()
    else:
        raise SystemExit("--qid 또는 --question-file 중 하나가 필요함 (한글 콘솔 인코딩 문제로 CLI 직접 인자는 지원 안 함)")

    run_id = new_run_id(question)
    run_metric = RunMetrics()

    t0 = time.monotonic()
    with run_metrics(run_metric):
        retrieval = run_retrieval(question, config, backend)
        t1 = time.monotonic()
        record_timing("retrieve", t1 - t0)

        answer_result = generate_answer(question, retrieval, backend, config)
        t2 = time.monotonic()
        record_timing("answer", t2 - t1)
        record_timing("total", t2 - t0)

    metrics_dict = run_metric.to_dict()

    final = build_final_json(
        question,
        run_id=run_id,
        config=config,
        selected_documents=retrieval.selected_documents,
        selected_pages=[
            {k: v for k, v in p.items() if k != "text"} for p in retrieval.selected_pages
        ],
        answer_path=answer_result["answer_path"],
        final_answer=answer_result["final_answer"],
        evidence=answer_result["evidence"],
        metrics=metrics_dict,
    )

    out_path = save_final_json(final, config, run_id)

    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"\n[저장됨] {out_path}")
    print(f"\n[총 소요시간] {metrics_dict['timings_seconds'].get('total', 0):.2f}초")
    print(f"[모델 호출수] embed={metrics_dict['embed_calls']} text_answer={metrics_dict['text_answer_calls']} vision_answer={metrics_dict['vision_answer_calls']}")

    if args.qid:
        item = get_eval_item(args.eval_file or _DEFAULT_EVAL_FILE, args.qid)
        expected_keywords = item.get("expected_answer_keywords", [])
        found = [kw for kw in expected_keywords if kw in answer_result["final_answer"]]
        missing = [kw for kw in expected_keywords if kw not in answer_result["final_answer"]]
        print(f"\n[기대 문서] {item.get('expected_documents')}")
        print(f"[기대 페이지] {item.get('expected_pages')}")
        print(f"[선정 문서] {[d['document_name'] for d in retrieval.selected_documents]}")
        print(f"[선정 페이지] {[p['page_number'] for p in retrieval.selected_pages]}")
        print(f"[답변 경로] {retrieval.answer_path} ({retrieval.route_reason})")
        print(f"[키워드 매칭] found={found} missing={missing}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .config import load_config
    from .eval_sets import load_eval_set
    from .evaluate import run_evaluation, save_evaluation
    from .models import get_backend

    config = load_config(args.config)
    config.ensure_dirs()
    backend = get_backend(config)

    eval_items = load_eval_set(args.eval_file or _DEFAULT_EVAL_FILE)
    if args.limit:
        eval_items = eval_items[: args.limit]

    result = run_evaluation(eval_items, config, backend)
    out_path = save_evaluation(result, config)

    for item in result["items"]:
        print(json.dumps(item, ensure_ascii=False))
    print("\n" + json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"\n[저장됨] {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag2")
    parser.add_argument("--config", default=None, help="config.yaml 경로 (기본: 패키지 내 config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="모델 호출 없이 인터프리터/패키지/torch CUDA/Ollama/컬렉션 count 점검")
    p_check.add_argument("--skip-indexes", action="store_true", help="Chroma 인덱스 count 점검 생략(ingest 전에는 항상 비어있음)")
    p_check.set_defaults(func=cmd_check)

    p_ingest = sub.add_parser("ingest", help="카탈로그+PDF 13개 ingest (MinerU 파싱 -> page_index/catalog_index 구축)")
    p_ingest.add_argument("--limit-docs", type=int, default=None, help="처리할 문서 수 제한 (smoke test용)")
    p_ingest.add_argument("--force", action="store_true", help="캐시(manifest) 무시하고 재파싱")
    p_ingest.set_defaults(func=cmd_ingest)

    p_add = sub.add_parser("add", help="신규 PDF 증분 추가 (선행: Excel 카탈로그 행 추가 + documents_dir에 PDF 복사)")
    p_add.add_argument("--pdf", action="append", default=None,
                       help="추가/교체할 PDF 파일명(카탈로그 매칭 기준). 여러 번 지정 가능")
    p_add.add_argument("--skip-vlm", action="store_true",
                       help="figure 페이지 vlm-engine 텍스트화(Phase 3 품질 단계) 생략")
    p_add.add_argument("--force-parse", action="store_true", help="파싱 캐시 무시하고 MinerU 재파싱")
    p_add.add_argument("--remove", default=None, help="문서를 인덱스에서 제거(파일명 또는 doc_slug)")
    p_add.set_defaults(func=cmd_add)

    p_ask = sub.add_parser("ask", help="질문 1건 답변 (한글 콘솔 인코딩 문제로 --qid 또는 --question-file 사용)")
    p_ask.add_argument("--qid", default=None, help="rag_eval_dataset.json의 id (예: core_001)")
    p_ask.add_argument("--question-file", default=None, help="UTF-8로 저장된 질문 텍스트 파일 경로")
    p_ask.add_argument("--eval-file", default=None, help="평가셋 JSON 경로 (기본: test_2/rag_eval_dataset.json)")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("evaluate", help="rag_eval_dataset.json으로 파이프라인 일괄 평가 (요약 리포트 1개만 저장)")
    p_eval.add_argument("--eval-file", default=None, help="평가셋 JSON 경로 (기본: test_2/rag_eval_dataset.json)")
    p_eval.add_argument("--limit", type=int, default=None, help="평가할 문항 수 제한 (dev용)")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

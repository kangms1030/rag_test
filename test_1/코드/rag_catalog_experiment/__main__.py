from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("rag_catalog_experiment")


def _parse_set_overrides(raw: list[str] | None) -> dict[str, object]:
    """`--set KEY=VALUE`(반복 가능)를 config.yaml 오버라이드 dict로 변환.

    load_config가 `bool(raw[...])`/`int(raw[...])`/`float(raw[...])`로 자체 캐스팅하는데,
    문자열 "false"는 `bool("false") == True`가 되는 함정이 있다. 여기서 미리 실제 타입으로
    파싱해 넘기면 그 함정을 피할 수 있다.
    """
    overrides: dict[str, object] = {}
    for entry in raw or []:
        if "=" not in entry:
            raise SystemExit(f"--set 형식 오류 (KEY=VALUE 필요): {entry!r}")
        key, _, value = entry.partition("=")
        key = key.strip()
        value = value.strip()
        if value.lower() in ("true", "false"):
            parsed: object = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        overrides[key] = parsed
    return overrides


#: config를 로드하는 서브커맨드가 공유하는 `--set KEY=VALUE` (반복 가능). 실험용 값을
#: config.yaml을 건드리지 않고 그때그때 덮어쓰기 위함 (예: --set index_dir=index/iso_catalog).
_set_parent = argparse.ArgumentParser(add_help=False)
_set_parent.add_argument(
    "--set",
    action="append",
    default=None,
    metavar="KEY=VALUE",
    help="config.yaml 값을 이번 실행에서만 덮어씀 (반복 가능)",
)


def cmd_ingest(args: argparse.Namespace) -> None:
    from .catalog import load_catalog, match_catalog_to_pdfs, save_match_report
    from .config import load_config
    from .indexes import get_index
    from .models import get_backend
    from .page_summary import ensure_page_summaries, seed_pending_pages
    from .pdf_parse import get_or_parse_document
    from .utils import doc_slug
    from .visual_chunk import ensure_visual_chunks

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()
    backend = get_backend(config, mock=args.mock)

    rows = load_catalog(config)
    report = match_catalog_to_pdfs(rows, config.documents_dir)
    save_match_report(report, config.output_dir / "catalog_match_report.json")

    catalog_index = get_index("catalog_index", config, backend)
    ids, texts, metas = [], [], []
    for row in rows:
        if not row.matched_file_path:
            continue
        ids.append(row.row_id)
        texts.append(row.catalog_search_text)
        metas.append(
            {
                "document_name": Path(row.matched_file_path).name,
                "file_path": row.matched_file_path,
                "doc_slug": doc_slug(row.matched_file_path),
                "title": row.columns.get("title", ""),
                "theme": row.columns.get("theme", ""),
                "publisher": row.columns.get("publisher", ""),
            }
        )
    catalog_index.upsert(ids, texts, metas)
    logger.info("catalog_index: %d개 row 색인 완료", len(ids))

    filename_index = get_index("filename_index", config, backend)
    fn_ids, fn_texts, fn_metas = [], [], []
    for row in rows:
        if not row.matched_file_path:
            continue
        rel = Path(row.matched_file_path)
        folder = rel.parent.name if rel.parent != Path(".") else ""
        search_text = f"{rel.stem} {folder}".strip()
        fn_ids.append(row.row_id)
        fn_texts.append(search_text)
        fn_metas.append(
            {
                "document_name": rel.name,
                "file_path": row.matched_file_path,
                "doc_slug": doc_slug(row.matched_file_path),
            }
        )
    filename_index.upsert(fn_ids, fn_texts, fn_metas)
    logger.info("filename_index: %d개 row 색인 완료 (파일명+폴더명만, filename_only 모드용)", len(fn_ids))

    matched_rows = [r for r in rows if r.matched_file_path]
    if args.limit_docs:
        matched_rows = matched_rows[: args.limit_docs]

    page_index = get_index("page_index", config, backend)
    chunk_index = get_index("visual_chunk_index", config, backend)

    total_pages = 0
    scanned_docs = 0
    for row in matched_rows:
        rel_path = row.matched_file_path
        abs_path = config.documents_dir / rel_path
        doc_info = get_or_parse_document(abs_path, rel_path, config, limit_pages=args.limit_pages, force=args.force)
        total_pages += doc_info.page_count
        if doc_info.is_scanned:
            scanned_docs += 1
        logger.info(
            "[%s] pages=%d text_ratio=%.2f scanned=%s",
            doc_info.document_name,
            doc_info.page_count,
            doc_info.text_page_ratio,
            doc_info.is_scanned,
        )
        n_seeded = seed_pending_pages(doc_info.pages, doc_info.doc_slug, page_index)
        logger.info("  page_index placeholder 시드: %d개 (VLM 호출 없음)", n_seeded)

        if args.vlm_summary:
            ensure_page_summaries(doc_info.pages, doc_info.doc_slug, backend, config, page_index)
            if args.visual_chunking:
                from .page_summary import get_or_build_summary

                summaries = {p.page_number: get_or_build_summary(p, doc_info.doc_slug, backend, config)[0] for p in doc_info.pages}
                ensure_visual_chunks(doc_info.pages, doc_info.doc_slug, summaries, backend, config, chunk_index)

    logger.info(
        "ingest 완료: 카탈로그 row %d개 중 %d개 매칭, 문서 %d개 파싱, 총 페이지 %d개 (스캔 문서 %d개)",
        len(rows),
        len(report.matched),
        len(matched_rows),
        total_pages,
        scanned_docs,
    )

    summary = {
        "mock": args.mock,
        "catalog_rows": len(rows),
        "catalog_matched": len(report.matched),
        "catalog_unmatched": len(report.unmatched_catalog_rows),
        "pdfs_unmatched": len(report.unmatched_pdfs),
        "documents_parsed": len(matched_rows),
        "total_pages": total_pages,
        "scanned_documents": scanned_docs,
        "vlm_summary_prewarmed": args.vlm_summary,
        "visual_chunking_prewarmed": args.visual_chunking,
    }
    with open(config.output_dir / "ingest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_ask(args: argparse.Namespace) -> None:
    from .answer import build_page_evidence, generate_answer, verify_answer
    from .config import load_config
    from .metrics import RunMetrics, run_metrics
    from .models import get_backend
    from .retrieval import run_retrieval
    from .schema import SchemaValidationError, build_final_json, save_final_json, validate_final_json
    from .utils import new_run_id

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()
    backend = get_backend(config, mock=args.mock)

    # Windows cmd 중첩 따옴표를 거치면 질문 앞뒤에 공백이 붙는 경우가 있다(run_id 해시와 로그를 오염시킴).
    args.question = args.question.strip()
    run_id = new_run_id(args.question)
    retrieval_mode = args.retrieval_mode or config.retrieval_mode

    run_metric = RunMetrics(mode=retrieval_mode, depth="answer")
    t0 = time.monotonic()
    with run_metrics(run_metric):
        retrieval = run_retrieval(
            args.question,
            config,
            backend,
            mode=retrieval_mode,
            top_docs=args.top_docs,
            top_pages=args.top_pages,
            top_chunks=args.top_chunks,
            limit_pages=args.limit_pages,
        )
        t1 = time.monotonic()
        answer_result = generate_answer(args.question, retrieval, backend, config)

        page_evidence = build_page_evidence(answer_result["raw_evidence"], answer_result["images_used"], retrieval.selected_documents, run_id, config)
        verification = verify_answer(args.question, answer_result["final_answer"], page_evidence, backend, config)
    t2 = time.monotonic()

    metrics_dict = run_metric.to_dict()
    metrics_dict.update(
        {
            "elapsed_seconds_total": round(t2 - t0, 3),
            "elapsed_seconds_retrieval": round(t1 - t0, 3),
            "elapsed_seconds_answer": round(t2 - t1, 3),
            "selected_doc_count": len(retrieval.selected_documents),
            "selected_page_count": len(retrieval.selected_pages),
            "selected_chunk_count": len(retrieval.selected_chunks),
            "evidence_count": len(page_evidence),
        }
    )

    final = build_final_json(
        args.question,
        retrieval,
        answer_result,
        page_evidence,
        verification,
        run_id=run_id,
        retrieval_mode=retrieval_mode,
        config=config,
        metrics=metrics_dict,
        mock=args.mock,
    )

    try:
        validate_final_json(final)
    except SchemaValidationError as e:
        logger.error("스키마 검증 실패: %s", e)
        final["run_status"] = "failed"
        final["schema_validation_error"] = str(e)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        out_path = save_final_json(final, config, run_id)

    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"\n[저장됨] {out_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .config import load_config
    from .eval_sets import load_eval_set
    from .evaluate import run_evaluation, save_evaluation
    from .models import get_backend

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()
    backend = get_backend(config, mock=args.mock)

    eval_items = load_eval_set(args.eval_file)
    retrieval_mode = args.retrieval_mode or config.retrieval_mode
    result = run_evaluation(eval_items, config, backend, limit=args.limit, retrieval_mode=retrieval_mode, depth=args.depth)
    save_evaluation(result, config)
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


def cmd_compare(args: argparse.Namespace) -> None:
    from .config import load_config
    from .eval_sets import load_eval_set
    from .experiments.compare import render_compare_markdown, run_compare, save_compare
    from .models import get_backend

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()
    backend = get_backend(config, mock=args.mock)

    eval_items = load_eval_set(args.eval_file)
    result = run_compare(eval_items, config, backend, modes=args.modes, limit=args.limit, depth=args.depth, stratify=args.stratify)
    json_path, md_path = save_compare(result, config)
    print(render_compare_markdown(result))
    print(f"\n[저장됨] {json_path}\n[저장됨] {md_path}")


def cmd_thresholds(args: argparse.Namespace) -> None:
    from .config import load_config
    from .eval_sets import load_eval_set
    from .experiments.thresholds import build_threshold_report, render_threshold_markdown, save_threshold_report
    from .models import get_backend

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()
    backend = get_backend(config, mock=args.mock)

    eval_items = load_eval_set(args.eval_file)
    if args.limit:
        eval_items = eval_items[: args.limit]
    report = build_threshold_report(eval_items, config, backend, modes=args.modes)
    json_path, md_path = save_threshold_report(report, config)
    print(render_threshold_markdown(report))
    print(f"\n[저장됨] {json_path}\n[저장됨] {md_path}")


def cmd_gen_eval(args: argparse.Namespace) -> None:
    from .config import load_config
    from .eval_sets import build_human_eval, build_synthetic_eval, save_eval_set

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()

    if args.mode == "synthetic":
        from .models import get_backend

        backend = get_backend(config, mock=args.mock)
        items = build_synthetic_eval(config, backend, n_per_doc=args.n_per_doc)
    else:
        items = build_human_eval(config, target_count=args.target_count, include_irrelevant=args.include_irrelevant)

    save_eval_set(items, Path(args.out))
    print(f"평가셋 {len(items)}개 생성: {args.out}")


def cmd_import_qa(args: argparse.Namespace) -> None:
    from .eval_sets import import_qa_xlsx, save_eval_set

    items = import_qa_xlsx(args.qa_file, sheet=args.sheet)
    save_eval_set(items, Path(args.out))
    print(f"QA 평가셋 {len(items)}개 임포트: {args.out}")


def cmd_check_env(args: argparse.Namespace) -> None:
    from .envcheck import check_environment

    config = None
    try:
        from .config import load_config

        config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    except Exception as e:
        logger.warning("config 로드 실패, ollama 서버/모델 점검은 생략: %s", e)

    result = check_environment(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_required_ok"]:
        raise SystemExit(1)


def cmd_clean_runs(args: argparse.Namespace) -> None:
    from .config import load_config
    from .runs import clean_runs

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()
    plans = clean_runs(config, dry_run=args.dry_run)
    if not plans:
        print("이동할 파일 없음")
        return
    for p in plans:
        prefix = "[dry-run] " if args.dry_run else ""
        print(f"{prefix}{p.src.name} -> runs/{p.category}/ ({p.reason})")


def cmd_validate_outputs(args: argparse.Namespace) -> None:
    from .config import load_config
    from .runs import validate_outputs, write_runs_md

    config = load_config(args.config, overrides=_parse_set_overrides(args.set))
    config.ensure_dirs()
    summary = validate_outputs(config)
    write_runs_md(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag_catalog_experiment")
    parser.add_argument("--config", default=None, help="config.yaml 경로 (기본: 패키지 내 config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="카탈로그+PDF ingest (기본: VLM 호출 없음)", parents=[_set_parent])
    p_ingest.add_argument("--limit-docs", type=int, default=None, help="처리할 문서 수 제한 (dev/smoke test용)")
    p_ingest.add_argument("--limit-pages", type=int, default=None, help="문서당 처리할 페이지 수 제한")
    p_ingest.add_argument("--vlm-summary", action="store_true", help="ingest 시점에 페이지 요약을 미리 생성 (기본은 lazy)")
    p_ingest.add_argument("--visual-chunking", action="store_true", help="--vlm-summary와 함께 visual chunk도 미리 생성")
    p_ingest.add_argument("--force", action="store_true", help="캐시 무시하고 재생성")
    p_ingest.add_argument("--mock", action="store_true", help="모델 호출 없이 MockBackend로 배선만 검증")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="질문에 답변", parents=[_set_parent])
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--retrieval-mode", choices=["catalog", "no_catalog", "filename_only"], default=None, help="기본: config.yaml의 retrieval_mode"
    )
    p_ask.add_argument("--top-docs", type=int, default=None)
    p_ask.add_argument("--top-pages", type=int, default=None)
    p_ask.add_argument("--top-chunks", type=int, default=None)
    p_ask.add_argument("--limit-pages", type=int, default=None, help="스캔 문서 전체 VLM 요약 시 페이지 상한")
    p_ask.add_argument("--mock", action="store_true")
    p_ask.add_argument("--out", default=None, help="결과 JSON 저장 경로 (기본: outputs/runs/{success,failed,mock}/answer_{run_id}.json)")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("evaluate", help="외부 평가셋으로 파이프라인 평가", parents=[_set_parent])
    p_eval.add_argument("--eval-file", required=True)
    p_eval.add_argument("--retrieval-mode", choices=["catalog", "no_catalog", "filename_only"], default=None)
    p_eval.add_argument("--depth", choices=["docs", "pages", "answer"], default="answer", help="docs/pages는 VLM 답변 생성을 건너뛰고 검색만 저렴하게 평가")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--mock", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_compare = sub.add_parser("compare", help="동일 평가셋을 여러 retrieval-mode로 돌려 비교 (compare_{ts}.json/.md)", parents=[_set_parent])
    p_compare.add_argument("--eval-file", required=True)
    p_compare.add_argument("--modes", nargs="+", choices=["catalog", "no_catalog", "filename_only"], default=["catalog", "no_catalog", "filename_only"])
    p_compare.add_argument("--depth", choices=["docs", "pages", "answer"], default="answer")
    p_compare.add_argument("--limit", type=int, default=None)
    p_compare.add_argument("--stratify", default=None, help="eval item 필드명(예: category)으로 추가 층화 리포트")
    p_compare.add_argument("--mock", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    p_thresh = sub.add_parser("thresholds", help="모드별 거절 게이트 dense 유사도 분포 리포트 (VLM 미호출)", parents=[_set_parent])
    p_thresh.add_argument("--eval-file", required=True)
    p_thresh.add_argument("--modes", nargs="+", choices=["catalog", "no_catalog", "filename_only"], default=["catalog", "no_catalog", "filename_only"])
    p_thresh.add_argument("--limit", type=int, default=None)
    p_thresh.add_argument("--mock", action="store_true")
    p_thresh.set_defaults(func=cmd_thresholds)

    p_gen = sub.add_parser("gen-eval", help="human(카탈로그 예상질문 기반, LLM 미사용) 또는 synthetic 평가셋 생성", parents=[_set_parent])
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--mode", choices=["human", "synthetic"], default="human")
    p_gen.add_argument("--target-count", type=int, default=20, help="human 모드: 총 문항 수 (문서별 배분 + 무관 질문)")
    p_gen.add_argument("--include-irrelevant", type=int, default=4, help="human 모드: 무관 질문 필러 개수")
    p_gen.add_argument("--n-per-doc", type=int, default=2, help="synthetic 모드: 문서당 질문 수")
    p_gen.add_argument("--mock", action="store_true")
    p_gen.set_defaults(func=cmd_gen_eval)

    p_qa = sub.add_parser("import-qa", help="QA 샘플 엑셀(장애 Q&A)을 sample_qa 평가셋으로 변환")
    p_qa.add_argument("--qa-file", required=True)
    p_qa.add_argument("--sheet", default="장애 Q&A 질답쌍")
    p_qa.add_argument("--out", required=True)
    p_qa.set_defaults(func=cmd_import_qa)

    p_env = sub.add_parser("check-env", help="모델 호출 없이 의존성/Ollama 서버/모델 설치 여부만 점검", parents=[_set_parent])
    p_env.set_defaults(func=cmd_check_env)

    p_clean = sub.add_parser("clean-runs", help="outputs/ 루트의 answer_*.json/evaluation_*.json을 분류 이동", parents=[_set_parent])
    p_clean.add_argument("--dry-run", action="store_true", help="이동 계획만 출력, 실제 이동 없음")
    p_clean.set_defaults(func=cmd_clean_runs)

    p_validate = sub.add_parser("validate-outputs", help="outputs/runs/success/*가 최신 schema를 만족하는지 검사, RUNS.md/latest/ 갱신", parents=[_set_parent])
    p_validate.set_defaults(func=cmd_validate_outputs)

    return parser


def main() -> None:
    # Windows 콘솔은 기본 cp949라 VLM/LLM이 생성한 텍스트(em-dash 등 cp949 밖 문자를
    # 흔히 포함)를 print()하면 UnicodeEncodeError로 죽는다. UTF-8로 강제하고, 그래도
    # 표현 못 하는 문자는 깨진 채로라도 계속 진행하게 한다(크래시보다 낫다).
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

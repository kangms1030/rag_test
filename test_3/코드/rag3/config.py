from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclasses.dataclass
class Config:
    catalog_excel_path: Path
    documents_dir: Path
    catalog_sheet: str

    output_dir: Path
    index_dir: Path
    cache_dir: Path

    parser: str
    mineru_device: str
    mineru_backend: str
    mineru_lang: str

    llm_provider: str
    text_answer_model: str
    vision_answer_model: str
    fallback_model: str

    embedding_model: str
    embed_doc_prefix: str
    embed_query_prefix: str

    top_docs: int
    top_pages: int

    min_doc_score: float
    doc_score_gap_ratio: float
    min_dense_similarity: float

    tokenizer: str

    figure_area_ratio_threshold: float
    scanned_table_verify: bool

    answer_image_dpi: int
    answer_image_max_side: int

    rrf_k: int

    ollama_num_ctx: int
    ollama_keep_alive: str
    ollama_num_predict: int

    config_path: Path

    #: 콜드스타트 whitespace 런어웨이 방어. done_reason=length(자연 종료 아님=미완/폭주)면
    #: 동일 호출을 1회 재발행(서버 KV 캐시 웜 -> 정상 완성). temp=0 그리디 유지(숫자 충실도 보존).
    ollama_retry_on_length: bool = True
    #: 문항당 length-retry 예산. 단일 콜드 폭주는 고치되, 병리적 문맥(재시도해도 계속 length인
    #: 지저분한 표)에서 여러 호출이 각각 재시도하며 지연이 폭발하는 것을 막는다(core_002: 4회 198s).
    ollama_max_length_retries: int = 2
    #: 재현성 고정 시드(temp=0 그리디에선 사실상 no-op이나 의도 문서화 + temp 변경 대비).
    ollama_seed: int = 0

    # === Phase 1 신규 (모두 기본값 보유; dataclass 규칙상 무기본값 필드 뒤에 위치) ===
    #: test_2가 이미 파싱한 캐시(manifest+content_list)를 청크화 소스로 재사용. None이면 parsed_dir 사용.
    source_parsed_dir: Path | None = None
    use_catalog_gate: bool = False
    chunk_target_chars: int = 700
    chunk_max_chars: int = 1200
    chunk_table_split_chars: int = 2500
    chunk_min_chars: int = 40
    retrieve_candidates: int = 20
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_max_length: int = 2048
    rerank_device: str = "cuda"
    rerank_score_floor: float = 0.0
    final_pages: int = 3
    # === Phase 1 검색 튜닝(ablation 토글) ===
    #: 리랭커에 카탈로그 프리픽스를 제거한 raw 청크 텍스트를 넣는다(cross-encoder 변별력 희석 방지).
    rerank_use_raw_text: bool = False
    #: 청크->페이지 집계 방식. "max"(최고 청크 점수) | "sum_topk"(best + decay*나머지 top-k).
    page_score_agg: str = "max"
    page_score_topk: int = 3
    page_score_decay: float = 0.5
    # === Phase 2: 검증 + 롤백 ===
    enable_verify: bool = True
    enable_rollback: bool = True
    enable_crag: bool = True
    context_max_chars: int = 10000
    crag_retry_floor: float = 0.02
    rollback_rerank_tau_high: float = 0.5
    deadline_seconds: int = 180
    verify_model: str = ""

    @property
    def source_parsed(self) -> Path:
        """청크화 소스 디렉터리. source_parsed_dir 지정 시 그걸, 아니면 자체 parsed_dir."""
        return self.source_parsed_dir if self.source_parsed_dir is not None else self.parsed_dir

    @property
    def parsed_dir(self) -> Path:
        return self.cache_dir / "parsed"

    @property
    def evidence_dir(self) -> Path:
        return self.output_dir / "evidence"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    def ensure_dirs(self) -> None:
        for d in (
            self.output_dir,
            self.index_dir,
            self.cache_dir,
            self.parsed_dir,
            self.evidence_dir,
            self.chroma_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    config_path = Path(path).resolve() if path else (_PACKAGE_DIR / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    base = config_path.parent
    return Config(
        catalog_excel_path=_resolve(base, raw["catalog_excel_path"]),
        documents_dir=_resolve(base, raw["documents_dir"]),
        catalog_sheet=raw["catalog_sheet"],
        output_dir=_resolve(base, raw["output_dir"]),
        index_dir=_resolve(base, raw["index_dir"]),
        cache_dir=_resolve(base, raw["cache_dir"]),
        parser=raw.get("parser", "mineru"),
        mineru_device=raw.get("mineru_device", "cuda"),
        mineru_backend=raw.get("mineru_backend", "pipeline"),
        mineru_lang=raw.get("mineru_lang", "korean"),
        llm_provider=raw["llm_provider"],
        text_answer_model=raw["text_answer_model"],
        vision_answer_model=raw["vision_answer_model"],
        fallback_model=raw["fallback_model"],
        embedding_model=raw["embedding_model"],
        embed_doc_prefix=raw["embed_doc_prefix"],
        embed_query_prefix=raw["embed_query_prefix"],
        top_docs=int(raw["top_docs"]),
        top_pages=int(raw["top_pages"]),
        min_doc_score=float(raw["min_doc_score"]),
        doc_score_gap_ratio=float(raw["doc_score_gap_ratio"]),
        min_dense_similarity=float(raw["min_dense_similarity"]),
        tokenizer=raw.get("tokenizer", "kiwi"),
        figure_area_ratio_threshold=float(raw.get("figure_area_ratio_threshold", 0.5)),
        scanned_table_verify=bool(raw.get("scanned_table_verify", True)),
        answer_image_dpi=int(raw.get("answer_image_dpi", 200)),
        answer_image_max_side=int(raw.get("answer_image_max_side", 2048)),
        rrf_k=int(raw["rrf_k"]),
        ollama_num_ctx=int(raw["ollama_num_ctx"]),
        ollama_keep_alive=raw["ollama_keep_alive"],
        ollama_num_predict=int(raw.get("ollama_num_predict", 0)),
        ollama_retry_on_length=bool(raw.get("ollama_retry_on_length", True)),
        ollama_max_length_retries=int(raw.get("ollama_max_length_retries", 2)),
        ollama_seed=int(raw.get("ollama_seed", 0)),
        config_path=config_path,
        source_parsed_dir=(_resolve(base, raw["source_parsed_dir"]) if raw.get("source_parsed_dir") else None),
        use_catalog_gate=bool(raw.get("use_catalog_gate", False)),
        chunk_target_chars=int(raw.get("chunk_target_chars", 700)),
        chunk_max_chars=int(raw.get("chunk_max_chars", 1200)),
        chunk_table_split_chars=int(raw.get("chunk_table_split_chars", 2500)),
        chunk_min_chars=int(raw.get("chunk_min_chars", 40)),
        retrieve_candidates=int(raw.get("retrieve_candidates", 20)),
        rerank_model=raw.get("rerank_model", "BAAI/bge-reranker-v2-m3"),
        rerank_max_length=int(raw.get("rerank_max_length", 2048)),
        rerank_device=raw.get("rerank_device", "cuda"),
        rerank_score_floor=float(raw.get("rerank_score_floor", 0.0)),
        final_pages=int(raw.get("final_pages", 3)),
        rerank_use_raw_text=bool(raw.get("rerank_use_raw_text", False)),
        page_score_agg=raw.get("page_score_agg", "max"),
        page_score_topk=int(raw.get("page_score_topk", 3)),
        page_score_decay=float(raw.get("page_score_decay", 0.5)),
        enable_verify=bool(raw.get("enable_verify", True)),
        enable_rollback=bool(raw.get("enable_rollback", True)),
        enable_crag=bool(raw.get("enable_crag", True)),
        context_max_chars=int(raw.get("context_max_chars", 10000)),
        crag_retry_floor=float(raw.get("crag_retry_floor", 0.02)),
        rollback_rerank_tau_high=float(raw.get("rollback_rerank_tau_high", 0.5)),
        deadline_seconds=int(raw.get("deadline_seconds", 180)),
        verify_model=raw.get("verify_model", "") or "",
    )

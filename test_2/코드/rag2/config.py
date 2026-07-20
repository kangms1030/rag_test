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

    config_path: Path

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
        config_path=config_path,
    )

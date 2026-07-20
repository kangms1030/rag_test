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

    llm_provider: str
    llm_model: str
    vlm_model: str
    fallback_model: str

    embedding_model: str
    embed_doc_prefix: str
    embed_query_prefix: str

    retrieval_mode: str

    top_docs: int
    top_pages: int
    top_chunks: int

    page_prefilter_topn: int
    chunk_pages_topk: int
    max_images_per_call: int
    image_max_side: int
    page_render_dpi: int

    enable_ocr: bool
    enable_visual_chunking: bool

    verify_with_crop_image: bool
    verify_max_crop_images: int
    tokenizer: str

    min_doc_score: float
    doc_score_gap_ratio: float
    min_dense_similarity: float
    min_filename_dense_similarity: float
    min_page_dense_similarity: float
    no_catalog_page_prefilter_topn: int

    scanned_text_ratio_threshold: float

    rrf_k: int

    ollama_num_ctx: int
    ollama_keep_alive: str

    config_path: Path

    @property
    def pages_dir(self) -> Path:
        return self.cache_dir / "pages"

    @property
    def summaries_dir(self) -> Path:
        return self.cache_dir / "summaries"

    @property
    def chunks_dir(self) -> Path:
        return self.cache_dir / "chunks"

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
            self.pages_dir,
            self.summaries_dir,
            self.chunks_dir,
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
        llm_provider=raw["llm_provider"],
        llm_model=raw["llm_model"],
        vlm_model=raw["vlm_model"],
        fallback_model=raw["fallback_model"],
        embedding_model=raw["embedding_model"],
        embed_doc_prefix=raw["embed_doc_prefix"],
        embed_query_prefix=raw["embed_query_prefix"],
        retrieval_mode=raw.get("retrieval_mode", "catalog"),
        top_docs=int(raw["top_docs"]),
        top_pages=int(raw["top_pages"]),
        top_chunks=int(raw["top_chunks"]),
        page_prefilter_topn=int(raw["page_prefilter_topn"]),
        chunk_pages_topk=int(raw["chunk_pages_topk"]),
        max_images_per_call=int(raw["max_images_per_call"]),
        image_max_side=int(raw["image_max_side"]),
        page_render_dpi=int(raw["page_render_dpi"]),
        enable_ocr=bool(raw["enable_ocr"]),
        enable_visual_chunking=bool(raw["enable_visual_chunking"]),
        verify_with_crop_image=bool(raw.get("verify_with_crop_image", False)),
        verify_max_crop_images=int(raw.get("verify_max_crop_images", 3)),
        tokenizer=raw.get("tokenizer", "char_bigram"),
        min_doc_score=float(raw["min_doc_score"]),
        doc_score_gap_ratio=float(raw["doc_score_gap_ratio"]),
        min_dense_similarity=float(raw["min_dense_similarity"]),
        min_filename_dense_similarity=float(raw.get("min_filename_dense_similarity", 0.30)),
        min_page_dense_similarity=float(raw.get("min_page_dense_similarity", 0.35)),
        no_catalog_page_prefilter_topn=int(raw.get("no_catalog_page_prefilter_topn", 24)),
        scanned_text_ratio_threshold=float(raw["scanned_text_ratio_threshold"]),
        rrf_k=int(raw["rrf_k"]),
        ollama_num_ctx=int(raw["ollama_num_ctx"]),
        ollama_keep_alive=raw["ollama_keep_alive"],
        config_path=config_path,
    )

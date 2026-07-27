"""chatbot_demo_v2 데이터 부트스트랩 (1회 실행).

test_3/코드/rag3 의 색인(index)과 파싱 캐시(cache/parsed_v25)를 v2 내부 ragdata/ 로 복사한다.
원본은 읽기만 하며 절대 수정하지 않는다. 복사 후 page_store 표본의 근거 이미지 경로가
v2 config 기준으로 해석되는지 검증한다(D3 — stale 절대경로 방어).

실행:
    <intern_chatbot python> chatbot_demo_v2/scripts/bootstrap_data.py [--force]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]          # chatbot_demo_v2
PROJECT_ROOT = PKG_ROOT.parent                          # 챗봇
SRC_RAG3 = PROJECT_ROOT / "test_3" / "코드" / "rag3"
RAGDATA = PKG_ROOT / "ragdata"

# (원본 하위경로, 대상 하위폴더명)
COPY_JOBS = [
    (SRC_RAG3 / "index", RAGDATA / "index"),
    (SRC_RAG3 / "cache" / "parsed_v25", RAGDATA / "parsed_v25"),
]


def _copytree(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            print(f"  [skip] 이미 존재: {dst}  (재복사하려면 --force)")
            return
        print(f"  [clean] 기존 삭제: {dst}")
        shutil.rmtree(dst)
    t0 = time.time()
    print(f"  [copy] {src}  ->  {dst}")
    shutil.copytree(src, dst)
    n = sum(1 for _ in dst.rglob("*") if _.is_file())
    print(f"  [done] {n} files, {time.time() - t0:.1f}s")


def _validate() -> int:
    """복사된 데이터가 v2 config 로 정상 로드/해석되는지 검증. 성공하면 0, 실패하면 1."""
    sys.path.insert(0, str(PKG_ROOT / "ragcore"))
    from rag3.config import load_config
    from rag3.answer import resolve_cached_path
    from rag3.page_store import load_page_store

    cfg = load_config(PKG_ROOT / "ragcore" / "rag3" / "config.yaml")
    print(f"\n[검증] index_dir={cfg.index_dir}")
    print(f"[검증] source_parsed={cfg.source_parsed}")

    store = load_page_store(cfg)
    print(f"[검증] page_store 로드: {len(store)} pages")
    if not store:
        print("[오류] page_store 가 비어 있습니다.")
        return 1

    # has_table 페이지 표본 20개의 page_image_path 해석 성공률
    sampled = [m for m in store.values()
               if str(m.get("meta", {}).get("has_table")) == "True"][:20]
    if not sampled:
        sampled = list(store.values())[:20]
    ok = 0
    for m in sampled:
        meta = m.get("meta", {})
        stored = meta.get("page_image_path", "")
        if resolve_cached_path(stored, cfg) is not None:
            ok += 1
    rate = ok / len(sampled) if sampled else 0.0
    print(f"[검증] 근거 이미지 경로 해석: {ok}/{len(sampled)} ({rate:.0%})")
    if rate < 0.90:
        print("[오류] 해석 성공률 90% 미만 — 경로/복사 확인 필요.")
        return 1
    print("[검증] 통과 ✓")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="v2 데이터 부트스트랩")
    ap.add_argument("--force", action="store_true", help="기존 ragdata 삭제 후 재복사")
    args = ap.parse_args()

    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    for src, dst in COPY_JOBS:
        if not src.exists():
            print(f"[오류] 원본 없음: {src}")
            return 1
    RAGDATA.mkdir(parents=True, exist_ok=True)
    print("=== 복사 ===")
    for src, dst in COPY_JOBS:
        _copytree(src, dst, args.force)

    return _validate()


if __name__ == "__main__":
    raise SystemExit(main())

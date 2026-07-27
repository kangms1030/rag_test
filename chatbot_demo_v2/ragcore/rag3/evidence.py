"""답변 근거 페이지 이미지 해석/내보내기.

controller가 evidence에 실어주는 page_image_path/table_crop_path는 파싱 캐시
재구성으로 stale일 수 있어, answer.resolve_cached_path로 실제 파일을 찾아
절대경로를 부착한다(resolve_evidence). export_evidence는 그 이미지를
outputs/evidence/<run_id>/ 아래 ascii-safe 파일명으로 복사해 CLI/웹이 한글·공백
경로 없이 바로 참조할 수 있게 한다. 기존 결과 dict의 키는 변경하지 않고
새 키(page_image_resolved/table_crop_resolved)만 추가한다.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from .answer import resolve_cached_path
from .config import Config

logger = logging.getLogger(__name__)


def resolve_evidence(result: dict[str, Any], config: Config) -> list[dict[str, Any]]:
    """result['evidence'] 각 항목에 page_image_resolved/table_crop_resolved
    (존재 확인된 절대경로 str, 없으면 None)를 추가하고 evidence 리스트를 반환."""
    evidence = result.get("evidence", [])
    for item in evidence:
        img = resolve_cached_path(item.get("page_image_path", ""), config)
        crop = resolve_cached_path(item.get("table_crop_path", ""), config)
        item["page_image_resolved"] = str(img) if img else None
        item["table_crop_resolved"] = str(crop) if crop else None
    return evidence


def export_evidence(result: dict[str, Any], config: Config, run_id: str,
                    out_dir: Path | None = None) -> list[dict[str, Any]]:
    """resolved 근거 이미지를 {out_dir or evidence_dir/run_id}/에 복사하고 목록을 반환.

    파일명은 ev{순위}_p{페이지:04d}[_table]{확장자} — URL/콘솔에서 안전한 ascii.
    같은 디렉터리에 manifest.json(원본 경로·문서명 매핑)을 함께 기록한다.
    """
    evidence = resolve_evidence(result, config)
    dest = Path(out_dir) if out_dir else (config.evidence_dir / run_id)
    dest.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    for rank, item in enumerate(evidence, start=1):
        page = int(item.get("page_number") or 0)
        for kind, key in (("page", "page_image_resolved"), ("table_crop", "table_crop_resolved")):
            src = item.get(key)
            if not src:
                continue
            suffix = Path(src).suffix or ".png"
            name = f"ev{rank}_p{page:04d}{'_table' if kind == 'table_crop' else ''}{suffix}"
            try:
                shutil.copy2(src, dest / name)
            except OSError as e:
                logger.warning("근거 이미지 복사 실패(%s): %s", src, e)
                continue
            files.append({
                "rank": rank,
                "kind": kind,
                "document_name": item.get("document_name", ""),
                "page_number": page,
                "file": str(dest / name),
                "source": src,
            })

    (dest / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "question": result.get("question", ""), "files": files},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return files

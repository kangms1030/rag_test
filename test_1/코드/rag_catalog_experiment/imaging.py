"""이미지 리사이즈 / bbox crop / 근거 하이라이트.

`vlm_test_v2.ipynb`의 `crop_evidence` / `highlight_on_page`를 재사용하되, 두 함수가
서로 다른(패딩 vs 비패딩) 좌표를 쓰던 불일치를 없애기 위해 crop과 highlight 모두
동일한 패딩된 좌표를 사용하도록 통일했다.

bbox 계약: normalized 0.0~1.0, top-left(x1,y1) -> bottom-right(x2,y2).
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw

_HIGHLIGHT_COLORS = [(255, 200, 0), (50, 180, 255), (80, 220, 120), (230, 90, 200), (255, 120, 80)]


class NormBBox(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float


def clamp_bbox(bbox: dict, padding: float = 0.0) -> NormBBox:
    x1 = max(0.0, float(bbox["x1"]) - padding)
    y1 = max(0.0, float(bbox["y1"]) - padding)
    x2 = min(1.0, float(bbox["x2"]) + padding)
    y2 = min(1.0, float(bbox["y2"]) + padding)
    return NormBBox(x1, y1, x2, y2)


def is_valid_bbox(bbox: dict, min_area: float = 0.005) -> bool:
    try:
        x1, y1, x2, y2 = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    return (x2 - x1) * (y2 - y1) >= min_area


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def crop_image(image_path: Path, bbox: dict, padding: float = 0.02) -> tuple[Image.Image, NormBBox]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        padded = clamp_bbox(bbox, padding)
        px1, py1 = int(padded.x1 * w), int(padded.y1 * h)
        px2, py2 = int(padded.x2 * w), int(padded.y2 * h)
        px2, py2 = max(px2, px1 + 1), max(py2, py1 + 1)
        crop = img.crop((px1, py1, px2, py2)).copy()
        return crop, padded


def highlight_boxes(
    image_path: Path, boxes: list[dict], *, labels: list[str] | None = None, padding: float = 0.02
) -> Image.Image:
    with Image.open(image_path) as base:
        base = base.convert("RGBA")
        w, h = base.size
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for i, bbox in enumerate(boxes):
            padded = clamp_bbox(bbox, padding)
            r, g, b = _HIGHLIGHT_COLORS[i % len(_HIGHLIGHT_COLORS)]
            x1, y1, x2, y2 = padded.x1 * w, padded.y1 * h, padded.x2 * w, padded.y2 * h
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 70))
            draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 230), width=3)
            label = labels[i] if labels and i < len(labels) else str(i + 1)
            draw.rectangle([x1, y1, x1 + 22, y1 + 18], fill=(r, g, b, 220))
            draw.text((x1 + 4, y1 + 2), label, fill=(0, 0, 0, 255))
        return Image.alpha_composite(base, overlay).convert("RGB")


def save_jpeg(img: Image.Image, path: Path, quality: int = 95) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=quality)
    return path

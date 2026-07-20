"""이미지 리사이즈 / bbox crop 유틸. MinerU가 내놓는 블록 bbox로 표/도표 크롭을 만드는 데 쓴다.

bbox 계약: normalized 0.0~1.0, top-left(x1,y1) -> bottom-right(x2,y2).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def crop_norm_bbox(image_path: Path, x1: float, y1: float, x2: float, y2: float, padding: float = 0.01) -> Image.Image:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        px1 = max(0.0, x1 - padding)
        py1 = max(0.0, y1 - padding)
        px2 = min(1.0, x2 + padding)
        py2 = min(1.0, y2 + padding)
        left, top = int(px1 * w), int(py1 * h)
        right, bottom = int(px2 * w), int(py2 * h)
        right, bottom = max(right, left + 1), max(bottom, top + 1)
        return img.crop((left, top, right, bottom)).copy()


def save_jpeg(img: Image.Image, path: Path, quality: int = 95) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=quality)
    return path

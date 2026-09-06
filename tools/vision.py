"""Deterministic, conservative image preparation for local P&ID inference."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


@dataclass(frozen=True)
class VisionImage:
    path: Path
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    original_bytes: int
    processed_bytes: int
    scale_factor: float
    processing_duration_ms: float
    margin_removed: bool

    def metadata(self) -> dict[str, int | float | bool]:
        return {
            "original_width": self.original_width, "original_height": self.original_height,
            "processed_width": self.processed_width, "processed_height": self.processed_height,
            "original_bytes": self.original_bytes, "processed_bytes": self.processed_bytes,
            "scale_factor": self.scale_factor,
            "processing_duration_ms": self.processing_duration_ms,
            "margin_removed": self.margin_removed,
        }


class VisionPreprocessor:
    """Prepare at most one high-quality image without semantic interpretation."""

    def __init__(self, *, target_long_side: int | None = None, max_long_side: int | None = None,
                 max_file_size: int | None = None) -> None:
        self.target_long_side = target_long_side or int(os.getenv("VISION_TARGET_LONG_SIDE", "1792"))
        self.max_long_side = max_long_side or int(os.getenv("VISION_MAX_LONG_SIDE", "2048"))
        self.max_file_size = max_file_size or int(os.getenv("VISION_MAX_FILE_SIZE", str(20 * 1024 * 1024)))

    @staticmethod
    def _content_box(image: Image.Image) -> tuple[int, int, int, int] | None:
        # Only trim clearly near-white outer margins; retain a safety border.
        gray = ImageOps.grayscale(image)
        threshold = 245
        mask = gray.point(lambda p: 255 if p < threshold else 0)
        box = mask.getbbox()
        if not box:
            return None
        left, top, right, bottom = box
        pad = max(8, min(image.size) // 100)
        left, top = max(0, left - pad), max(0, top - pad)
        right, bottom = min(image.width, right + pad), min(image.height, bottom + pad)
        # Avoid risky crops when the detected content is already most of frame.
        if (right - left) * (bottom - top) < image.width * image.height * 0.55:
            return left, top, right, bottom
        return None

    def process(self, source: str | Path, destination: str | Path) -> VisionImage:
        started = time.perf_counter()
        source_path, destination_path = Path(source), Path(destination)
        if not source_path.is_file() or source_path.stat().st_size > self.max_file_size:
            raise ValueError("Image is missing or exceeds the configured size limit")
        with Image.open(source_path) as opened:
            opened.load()
            original_width, original_height = opened.size
            image = ImageOps.exif_transpose(opened).convert("RGB")
            box = self._content_box(image)
            margin_removed = box is not None
            if box:
                image = image.crop(box)
            longest = max(image.size)
            limit = self.max_long_side if longest > self.max_long_side else self.target_long_side
            if longest > limit:
                scale = limit / longest
                image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
            else:
                scale = 1.0
            # Mild enhancement only; preserve thin lines and text.
            image = ImageEnhance.Contrast(image).enhance(1.05)
            image = image.filter(ImageFilter.UnsharpMask(radius=0.6, percent=70, threshold=3))
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination_path, format="PNG", optimize=True)
            processed_bytes = destination_path.stat().st_size
            return VisionImage(destination_path, original_width, original_height, image.width, image.height,
                               source_path.stat().st_size, processed_bytes, round(scale, 4),
                               round((time.perf_counter() - started) * 1000, 2), margin_removed)


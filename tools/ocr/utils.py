"""Input validation and data normalization for OCR backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from runtime.errors import InvalidOCRInputError


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def validate_image_source(source: str | Path) -> Path:
    """Validate the initial image-only OCR input contract."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise InvalidOCRInputError(f"OCR image file not found: {path}")
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise InvalidOCRInputError(
            f"Unsupported OCR input '{path.suffix or '<no extension>'}'; "
            "supported formats: PNG, JPG, JPEG"
        )
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidOCRInputError(f"Invalid OCR image file: {path}") from exc
    return path


def plain_value(value: Any) -> Any:
    """Convert numpy/Paddle scalar or array values to JSON-safe Python data."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value

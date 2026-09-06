"""Local, backend-agnostic OCR capability."""

import json

from langchain_core.tools import tool

from .base import OCRBackend
from .models import OCRMetadata, OCRResult, OCRSource, OCRTextBlock
from .paddle import PaddleOCRBackend


@tool
def extract_ocr(image_path: str) -> str:
    """Extract local OCR text, bounding boxes, and confidence from a PNG/JPEG image.

    This explicit tool is registered for future document/image workflows; image
    routing remains unchanged and continues to select the vision model.
    """
    result = PaddleOCRBackend().extract(image_path)
    return json.dumps(result.model_dump(mode="json"), ensure_ascii=False)

__all__ = [
    "OCRBackend",
    "OCRMetadata",
    "OCRResult",
    "OCRSource",
    "OCRTextBlock",
    "PaddleOCRBackend",
    "extract_ocr",
]

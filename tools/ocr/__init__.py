"""Local, backend-agnostic OCR capability."""

import json

from langchain_core.tools import tool

from .base import OCRBackend
from .models import OCRMetadata, OCRResult, OCRSource, OCRTextBlock
from .paddle import PaddleOCRBackend
from runtime.extraction import ExtractedValue, ExtractionReviewQueue


@tool
def extract_ocr(image_path: str) -> str:
    """Extract local OCR text, bounding boxes, and confidence from a PNG/JPEG image.

    This explicit tool is registered for future document/image workflows; image
    routing remains unchanged and continues to select the vision model.
    """
    result = PaddleOCRBackend().extract(image_path)
    queue = ExtractionReviewQueue(threshold=0.85)
    review_items = []
    for block in result.text_blocks:
        item = ExtractedValue(value=block.text, confidence=block.confidence,
                              source=str(result.source.path), page=block.page_number,
                              bounding_box=tuple(block.bounding_box[0] + block.bounding_box[2]) if len(block.bounding_box) >= 3 else None,
                              method=result.metadata.backend)
        if queue.add(item):
            review_items.append(item.to_dict())
    payload = result.model_dump(mode="json")
    payload["confidence_threshold"] = queue.threshold
    payload["review_required"] = bool(review_items)
    payload["review_queue"] = review_items
    return json.dumps(payload, ensure_ascii=False)

__all__ = [
    "OCRBackend",
    "OCRMetadata",
    "OCRResult",
    "OCRSource",
    "OCRTextBlock",
    "PaddleOCRBackend",
    "extract_ocr",
]

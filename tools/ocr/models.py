"""Serialization-safe models returned by OCR backends."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


BoundingBox = list[list[float]]


class OCRSource(BaseModel):
    """Origin of an OCR result."""

    path: str
    media_type: str
    page_number: int = 1


class OCRTextBlock(BaseModel):
    """One detected text region and its geometry."""

    text: str
    bounding_box: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    page_number: int = 1


class OCRMetadata(BaseModel):
    """Backend and timing information for an OCR operation."""

    backend: str
    engine_version: str | None = None
    model: str | None = None
    device: str | None = None
    initialization_ms: float | None = None
    inference_ms: float | None = None


class OCRResult(BaseModel):
    """Normalized OCR output suitable for JSON and LangGraph state."""

    source: OCRSource
    text: str
    text_blocks: list[OCRTextBlock] = Field(default_factory=list)
    metadata: OCRMetadata
    input_type: Literal["image", "document"] = "image"

    @property
    def region_count(self) -> int:
        return len(self.text_blocks)

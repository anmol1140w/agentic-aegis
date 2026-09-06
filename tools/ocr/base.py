"""Backend interface for local OCR implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from .models import OCRResult


class OCRBackend(ABC):
    """Extract normalized OCR data without exposing a vendor API upstream."""

    @abstractmethod
    def extract(self, source: str | Path) -> OCRResult:
        """Extract OCR text, boxes, and confidences from one supported source."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether this backend can be initialized locally."""

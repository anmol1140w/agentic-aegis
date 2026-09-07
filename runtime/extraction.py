"""Evidence-bearing extraction primitives for industrial documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExtractedValue:
    value: Any
    confidence: float
    source: str
    page: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    method: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExtractionReviewQueue:
    """Small deterministic review queue; callers can persist its dictionaries."""

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self._items: list[dict[str, Any]] = []

    def add(self, item: ExtractedValue) -> bool:
        if item.confidence < self.threshold:
            self._items.append(item.to_dict())
            return True
        return False

    def pending(self) -> list[dict[str, Any]]:
        return list(self._items)

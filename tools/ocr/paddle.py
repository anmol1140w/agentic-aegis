"""PaddleOCR 3.x backend using its local ``PaddleOCR.predict`` API."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from runtime.errors import OCRBackendError

from .base import OCRBackend
from .models import OCRMetadata, OCRResult, OCRSource, OCRTextBlock
from .utils import plain_value, validate_image_source


class PaddleOCRBackend(OCRBackend):
    """Lazy local PaddleOCR backend for PNG/JPEG image recognition.

    The PaddleOCR import and model initialization are intentionally deferred to
    first use: launching the normal workbench does not load OCR models or make
    an OCR-specific network request.  Once PaddleOCR's models are cached,
    inference uses its local model files.
    """

    backend_name = "paddleocr"

    def __init__(
        self,
        *,
        language: str = "en",
        engine_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.language = language
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._initialization_ms: float | None = None
        self._engine_version: str | None = None
        self._device: str | None = None

    def health_check(self) -> bool:
        try:
            self._ensure_engine()
        except OCRBackendError:
            return False
        return True

    def extract(self, source: str | Path) -> OCRResult:
        path = validate_image_source(source)
        engine = self._ensure_engine()
        started = time.perf_counter()
        try:
            pages = list(engine.predict(str(path)))
        except Exception as exc:
            raise OCRBackendError(f"PaddleOCR inference failed for {path}: {exc}") from exc
        inference_ms = round((time.perf_counter() - started) * 1000, 2)

        blocks: list[OCRTextBlock] = []
        for page_number, page in enumerate(pages, start=1):
            mapping = self._page_mapping(page)
            texts = list(plain_value(mapping.get("rec_texts", [])) or [])
            scores = list(plain_value(mapping.get("rec_scores", [])) or [])
            boxes = list(plain_value(mapping.get("rec_boxes", [])) or [])
            for index, raw_text in enumerate(texts):
                text = str(plain_value(raw_text)).strip()
                if not text:
                    continue
                if index >= len(boxes):
                    continue
                confidence = float(plain_value(scores[index])) if index < len(scores) else 0.0
                blocks.append(OCRTextBlock(
                    text=text,
                    bounding_box=self._normalize_box(plain_value(boxes[index])),
                    confidence=max(0.0, min(1.0, confidence)),
                    page_number=page_number,
                ))

        return OCRResult(
            source=OCRSource(path=str(path), media_type=self._media_type(path)),
            text="\n".join(block.text for block in blocks),
            text_blocks=blocks,
            metadata=OCRMetadata(
                backend=self.backend_name,
                engine_version=self._engine_version,
                model=f"PaddleOCR lang={self.language}",
                device=self._device,
                initialization_ms=self._initialization_ms,
                inference_ms=inference_ms,
            ),
        )

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        started = time.perf_counter()
        try:
            if self._engine_factory is not None:
                self._engine = self._engine_factory(lang=self.language)
            else:
                import paddle
                import paddleocr
                from paddleocr import PaddleOCR

                self._engine_version = getattr(paddleocr, "__version__", None)
                self._device = paddle.device.get_device()
                self._engine = PaddleOCR(lang=self.language)
        except Exception as exc:
            raise OCRBackendError(f"Unable to initialize local PaddleOCR backend: {exc}") from exc
        self._initialization_ms = round((time.perf_counter() - started) * 1000, 2)
        return self._engine

    @staticmethod
    def _page_mapping(page: Any) -> Mapping[str, Any]:
        if isinstance(page, Mapping):
            return page
        data = getattr(page, "json", None)
        if isinstance(data, Mapping):
            return data
        if callable(data):
            value = data()
            if isinstance(value, Mapping):
                return value
        raise OCRBackendError("PaddleOCR returned an unsupported result format")

    @staticmethod
    def _media_type(path: Path) -> str:
        return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    @staticmethod
    def _normalize_box(box: Any) -> list[list[float]]:
        """Normalize Paddle's rectangle or polygon boxes to polygon points."""
        values = plain_value(box)
        if len(values) == 4 and all(not isinstance(item, (list, tuple)) for item in values):
            left, top, right, bottom = (float(item) for item in values)
            return [[left, top], [right, top], [right, bottom], [left, bottom]]
        return [[float(value) for value in point] for point in values]

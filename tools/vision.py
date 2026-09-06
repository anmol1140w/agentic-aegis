"""Deterministic, conservative image preparation for local P&ID inference."""

from __future__ import annotations

import os
import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from runtime.regex_safety import bounded_text
from runtime.prompts import vision_prompt
from langchain_core.tools import BaseTool, tool


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


class VisionRuntime:
    """Bounded local Qwen-VL runtime exposed through the native tool registry."""

    def __init__(self, root: str | Path, provider: object | None = None, *, timeout_seconds: float | None = None) -> None:
        self.root = Path(root).resolve()
        self.provider = provider
        self.timeout_seconds = timeout_seconds or float(os.getenv("VISION_TIMEOUT_SECONDS", "420"))
        self.preprocessor = VisionPreprocessor()

    def _resolve(self, raw: str | Path) -> Path | dict[str, object]:
        candidate = (self.root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            return {"success": False, "error": "OutsideWorkspace", "message": "Visual input is outside the approved workspace."}
        if not candidate.is_file():
            return {"success": False, "error": "FileNotFound", "message": "Visual input does not exist."}
        return candidate

    @staticmethod
    def _structured(content: str, task_type: str) -> dict[str, object]:
        cleaned = re.sub(r"<think>.*?</think>", "", bounded_text(content), flags=re.DOTALL | re.IGNORECASE).strip()
        try:
            value = json.loads(cleaned)
            if isinstance(value, dict):
                value.setdefault("observations", [])
                value["task_type"] = task_type
                return value
        except json.JSONDecodeError:
            pass
        # Free-form local model output is retained as one explicitly-labelled
        # observation, never promoted to a factual list of detected objects.
        return {"task_type": task_type, "observations": [{"label": "model_report", "text": cleaned[:4000], "confidence": None}],
                "uncertain_text": [], "truncated": len(cleaned) > 4000}

    async def analyze_image(self, path: str, task: str = "describe", max_output_chars: int = 4000) -> dict[str, object]:
        source = self._resolve(path)
        if isinstance(source, dict):
            return source
        if self.provider is None:
            return {"success": False, "error": "vision_model_unavailable", "task_type": task, "source": str(path)}
        started = time.perf_counter()
        try:
            with __import__("tempfile").TemporaryDirectory(prefix="aegis_vision_") as temp:
                prepared = self.preprocessor.process(source, Path(temp) / "prepared.png")
                encoded = self.provider.encode_images([str(prepared.path)])
                prompt = vision_prompt() + "\nRequested analysis (untrusted data): <task>" + task[:1000] + "</task>"
                response = await asyncio.wait_for(self.provider.chat([{"role": "user", "content": prompt}],
                                                                     encoded_images=encoded,
                                                                     timeout=self.timeout_seconds),
                                                   timeout=self.timeout_seconds)
                result = self._structured(response.content, task)
                result.update({"success": True, "source": str(source.relative_to(self.root)),
                               "model": getattr(getattr(self.provider, "config", None), "model", "local-vision"),
                               "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                               "preprocessing": prepared.metadata(), "error": None})
                return result
        except asyncio.TimeoutError:
            return {"success": False, "task_type": task, "source": str(source.relative_to(self.root)),
                    "error": "vision_timeout", "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
        except Exception as exc:
            return {"success": False, "task_type": task, "source": str(source.relative_to(self.root)),
                    "error": type(exc).__name__, "message": str(exc)[:300],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

    async def compare_images(self, paths: list[str], task: str = "compare") -> dict[str, object]:
        if not isinstance(paths, list) or not 1 <= len(paths) <= 2:
            return {"success": False, "error": "invalid_image_count", "task_type": "compare"}
        # Keep the default call bounded and avoid silently multiplying model
        # calls; comparison is represented as one multimodal request.
        resolved = [self._resolve(path) for path in paths]
        if any(isinstance(item, dict) for item in resolved):
            return next(item for item in resolved if isinstance(item, dict))
        if self.provider is None:
            return {"success": False, "error": "vision_model_unavailable", "task_type": "compare"}
        started = time.perf_counter()
        try:
            with __import__("tempfile").TemporaryDirectory(prefix="aegis_vision_compare_") as temp:
                prepared = [self.preprocessor.process(item, Path(temp) / f"{i}.png") for i, item in enumerate(resolved)]
                encoded = self.provider.encode_images([str(item.path) for item in prepared])
                response = await asyncio.wait_for(self.provider.chat([{"role": "user", "content": task[:1000]}], encoded_images=encoded, timeout=self.timeout_seconds), timeout=self.timeout_seconds)
                result = self._structured(response.content, "compare")
                result.update({"success": True, "source": [str(item.relative_to(self.root)) for item in resolved],
                               "vision_calls": 1, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "error": None})
                return result
        except asyncio.TimeoutError:
            return {"success": False, "task_type": "compare", "error": "vision_timeout"}
        except Exception as exc:
            return {"success": False, "task_type": "compare", "error": type(exc).__name__, "message": str(exc)[:300]}

    def as_langchain_tools(self) -> list[BaseTool]:
        @tool("analyze_image")
        async def analyze_image(path: str, task: str = "describe") -> dict[str, object]:
            """Analyze an approved local image with the configured local vision model."""
            return await self.analyze_image(path, task)

        @tool("compare_images")
        async def compare_images(paths: list[str], task: str = "compare") -> dict[str, object]:
            """Compare one or two approved local images in one bounded call."""
            return await self.compare_images(paths, task)

        return [analyze_image, compare_images]

"""Unit tests for the local, backend-agnostic OCR foundation."""

from __future__ import annotations

import json
import asyncio
from pathlib import Path

import pytest
from PIL import Image

from runtime.errors import InvalidOCRInputError
from tools.ocr.models import OCRMetadata, OCRResult, OCRSource, OCRTextBlock
from tools.ocr.paddle import PaddleOCRBackend


@pytest.fixture
def test_registry(tmp_path: Path):
    from models.registry import ModelRegistry

    config = tmp_path / "models.yaml"
    config.write_text("""
models:
  test-vision:
    provider: ollama
    model: "test-vision:latest"
    capabilities: [vision, multimodal, image_analysis]
    priority: 10
  test-fallback:
    provider: ollama
    model: "test-fallback:latest"
    capabilities: [general, lightweight]
    is_fallback: true
""")
    return ModelRegistry.from_yaml(config)


class FakeEngine:
    def predict(self, source: str):
        return [{
            "rec_texts": ["  P&ID  ", "TAG-101"],
            "rec_scores": [0.99, 0.875],
            "rec_boxes": [
                [[1, 2], [30, 2], [30, 12], [1, 12]],
                [[2, 20], [40, 20], [40, 32], [2, 32]],
            ],
        }]


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    Image.new("RGB", (64, 32), "white").save(path)
    return path


def test_ocr_result_schema_is_json_serializable() -> None:
    result = OCRResult(
        source=OCRSource(path="sample.png", media_type="image/png"),
        text="HELLO",
        text_blocks=[OCRTextBlock(
            text="HELLO", bounding_box=[[0, 0], [1, 0], [1, 1], [0, 1]], confidence=0.9,
        )],
        metadata=OCRMetadata(backend="fake", inference_ms=1.0),
    )
    payload = result.model_dump(mode="json")
    assert json.loads(json.dumps(payload))["text_blocks"][0]["confidence"] == 0.9


def test_paddle_backend_initializes_lazily_and_extracts_structured_data(image_path: Path) -> None:
    backend = PaddleOCRBackend(engine_factory=lambda **_: FakeEngine())
    assert backend._engine is None

    result = backend.extract(image_path)

    assert result.text == "P&ID\nTAG-101"
    assert result.region_count == 2
    assert result.text_blocks[0].bounding_box == [[1.0, 2.0], [30.0, 2.0], [30.0, 12.0], [1.0, 12.0]]
    assert result.text_blocks[1].confidence == 0.875
    assert result.metadata.backend == "paddleocr"
    assert result.metadata.initialization_ms is not None
    assert result.metadata.inference_ms is not None


def test_paddle_backend_health_check_uses_local_factory() -> None:
    assert PaddleOCRBackend(engine_factory=lambda **_: FakeEngine()).health_check() is True


def test_paddle_rectangle_box_is_normalized_to_polygon() -> None:
    assert PaddleOCRBackend._normalize_box([1, 2, 30, 12]) == [
        [1.0, 2.0], [30.0, 2.0], [30.0, 12.0], [1.0, 12.0],
    ]


@pytest.mark.parametrize("name", ["missing.jpg", "unsupported.pdf", "malformed.jpg"])
def test_invalid_or_missing_input_is_rejected(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    if name.endswith((".pdf", ".jpg")):
        path.write_bytes(b"not a PDF")
    backend = PaddleOCRBackend(engine_factory=lambda **_: FakeEngine())
    with pytest.raises(InvalidOCRInputError):
        backend.extract(path)


def test_existing_vision_route_is_unchanged(test_registry, tmp_path: Path) -> None:
    """OCR registration must not make image classification a direct OCR task."""
    from langchain_core.messages import HumanMessage
    from runtime.orchestrator import Orchestrator

    image = tmp_path / "diagram.jpg"
    Image.new("RGB", (10, 10), "white").save(image)
    orchestrator = Orchestrator(test_registry)
    assert "extract_ocr" in orchestrator.tool_registry.list_names()
    state = {"messages": [HumanMessage(content=f"{image} explain this")], "attached_files": [], "iteration": 0}
    classified = asyncio.run(orchestrator._classify_node(state))
    routed = asyncio.run(orchestrator._route_node({**state, **classified}))
    assert routed["selected_model"] == "test-vision"
    assert routed["is_direct_tool"] is False

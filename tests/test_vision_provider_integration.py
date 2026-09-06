"""Live, provider-level vision acceptance test (no CLI, router, or LangGraph)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.registry import ModelRegistry


@pytest.mark.asyncio
@pytest.mark.live_ollama
async def test_qwen_vision_receives_image_and_returns_visible_text() -> None:
    """Verify the exact multimodal request path against local Ollama.

    Run deliberately, because loading/generating on the local vision model can
    take minutes: ``RUN_LIVE_OLLAMA=1 pytest -m live_ollama -q``.
    """
    if os.environ.get("RUN_LIVE_OLLAMA") != "1":
        pytest.skip("Set RUN_LIVE_OLLAMA=1 to run the local Ollama acceptance test")

    root = Path(__file__).resolve().parents[1]
    image = root / "images__val__269.jpg"
    assert image.is_file() and image.stat().st_size > 0

    provider = ModelRegistry.from_yaml(root / "config" / "models.yaml").get_provider("qwen-vision")
    encoded = provider.encode_images([str(image)])
    # The provider may safely downscale a large source image before base64 encoding.
    assert encoded and len(encoded[0]) > 100

    parts = [
        token async for token in provider.stream_chat(
            [{"role": "user", "content": "Describe this image briefly."}],
            encoded_images=encoded,
            timeout=300.0,
        )
    ]
    assert "".join(parts).strip(), "Ollama returned an empty visible response"

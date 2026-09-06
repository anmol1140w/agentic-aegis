"""Tests for model registry and provider abstraction."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from PIL import Image

from models.base import ModelCapability, ModelConfig, ModelProvider
from models.ollama import OllamaProvider
from runtime.errors import InvalidImageError, ModelTimeoutError
from models.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Test YAML-driven model registry."""

    @pytest.fixture
    def yaml_path(self, tmp_path: Path) -> Path:
        content = """
models:
  alpha:
    provider: ollama
    model: "alpha:latest"
    capabilities: [general, reasoning]
    context_length: 16384
    priority: 8

  beta:
    provider: ollama
    model: "beta:latest"
    capabilities: [coding, debugging]
    context_length: 32768
    priority: 10

  gamma:
    provider: ollama
    model: "gamma:latest"
    capabilities: [general, lightweight]
    context_length: 4096
    priority: 1
    is_fallback: true
"""
        p = tmp_path / "models.yaml"
        p.write_text(content)
        return p

    def test_loads_all_models(self, yaml_path: Path) -> None:
        reg = ModelRegistry.from_yaml(yaml_path)
        assert set(reg.provider_names) == {"alpha", "beta", "gamma"}

    def test_get_provider_returns_correct_type(self, yaml_path: Path) -> None:
        reg = ModelRegistry.from_yaml(yaml_path)
        provider = reg.get_provider("alpha")
        assert isinstance(provider, OllamaProvider)
        assert provider.model_id == "alpha:latest"

    def test_get_provider_unknown_raises(self, yaml_path: Path) -> None:
        reg = ModelRegistry.from_yaml(yaml_path)
        with pytest.raises(KeyError, match="not_real"):
            reg.get_provider("not_real")

    def test_find_by_capability(self, yaml_path: Path) -> None:
        reg = ModelRegistry.from_yaml(yaml_path)
        coders = reg.find_by_capability(ModelCapability.CODING)
        assert len(coders) == 1
        assert coders[0].name == "beta"

    def test_find_by_multiple_capabilities(self, yaml_path: Path) -> None:
        reg = ModelRegistry.from_yaml(yaml_path)
        result = reg.find_by_capability(
            ModelCapability.CODING, ModelCapability.DEBUGGING,
        )
        assert len(result) == 1
        assert result[0].name == "beta"

    def test_get_fallback(self, yaml_path: Path) -> None:
        reg = ModelRegistry.from_yaml(yaml_path)
        fb = reg.get_fallback()
        assert fb is not None
        assert fb.name == "gamma"

    def test_list_models(self, yaml_path: Path) -> None:
        reg = ModelRegistry.from_yaml(yaml_path)
        configs = reg.list_models()
        assert len(configs) == 3

    def test_missing_yaml_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ModelRegistry.from_yaml(tmp_path / "nonexistent.yaml")

    def test_unknown_provider_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("""
models:
  m1:
    provider: fake_provider
    model: "fake:1b"
    capabilities: [general]
""")
        with pytest.raises(ValueError, match="Unknown provider"):
            ModelRegistry.from_yaml(bad_yaml)


# ---------------------------------------------------------------------------
# ModelConfig tests
# ---------------------------------------------------------------------------

class TestModelConfig:

    def test_defaults(self) -> None:
        cfg = ModelConfig(
            name="test",
            provider="ollama",
            model="test:1b",
            capabilities=[ModelCapability.GENERAL],
        )
        assert cfg.context_length == 8192
        assert cfg.priority == 5
        assert cfg.is_fallback is False

    def test_capability_enum(self) -> None:
        assert ModelCapability("coding") == ModelCapability.CODING
        assert ModelCapability("vision") == ModelCapability.VISION


# ---------------------------------------------------------------------------
# OllamaProvider unit tests (no actual Ollama needed)
# ---------------------------------------------------------------------------

class TestOllamaProvider:

    @pytest.mark.asyncio
    async def test_stream_total_deadline_cancels_hanging_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class HangingResponse:
            closed = False
            async def __aenter__(self): return self
            async def __aexit__(self, *args): self.closed = True
            def raise_for_status(self): pass
            async def aiter_lines(self):
                await asyncio.Event().wait()
                yield ""
        class HangingClient:
            def __init__(self): self.response = HangingResponse()
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def stream(self, *args, **kwargs): return self.response
        client = HangingClient()
        monkeypatch.setattr("models.ollama.httpx.AsyncClient", lambda **kwargs: client)
        cfg = ModelConfig(name="vision", provider="ollama", model="vision:latest", capabilities=[ModelCapability.VISION])

        started = time.perf_counter()
        with pytest.raises(ModelTimeoutError, match="deadline"):
            _ = [token async for token in OllamaProvider(cfg).stream_chat([{"role": "user", "content": "hi"}], timeout=0.02)]

        assert time.perf_counter() - started < 0.5
        assert client.response.closed is True

    @pytest.mark.asyncio
    async def test_stream_chat_events_separates_thinking_and_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Response:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def raise_for_status(self): pass
            async def aiter_lines(self):
                yield '{"message":{"thinking":"planning"}}'
                yield '{"message":{"content":"answer"},"done":true}'
        class Client:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def stream(self, *args, **kwargs): return Response()
        monkeypatch.setattr("models.ollama.httpx.AsyncClient", lambda **kwargs: Client())
        cfg = ModelConfig(name="general", provider="ollama", model="general:latest", capabilities=[ModelCapability.GENERAL])
        events = [event async for event in OllamaProvider(cfg).stream_chat_events(
            [{"role": "user", "content": "hi"}], think=True, timeout=1
        )]
        assert events == [{"kind": "thinking", "token": "planning"}, {"kind": "content", "token": "answer"}]

    def test_creates_provider(self) -> None:
        cfg = ModelConfig(
            name="test-ollama",
            provider="ollama",
            model="test:1b",
            capabilities=[ModelCapability.GENERAL],
        )
        provider = OllamaProvider(cfg)
        assert provider.name == "test-ollama"
        assert provider.model_id == "test:1b"

    def test_get_chat_model_returns_instance(self) -> None:
        cfg = ModelConfig(
            name="test-ollama",
            provider="ollama",
            model="test:1b",
            capabilities=[ModelCapability.GENERAL],
        )
        provider = OllamaProvider(cfg)
        model = provider.get_chat_model(temperature=0.5)
        # Should return a ChatOllama instance
        assert model is not None
        # Same temperature should return cached instance
        model2 = provider.get_chat_model(temperature=0.5)
        assert model is model2

    def test_different_temperature_gives_new_model(self) -> None:
        cfg = ModelConfig(
            name="test-ollama",
            provider="ollama",
            model="test:1b",
            capabilities=[ModelCapability.GENERAL],
        )
        provider = OllamaProvider(cfg)
        m1 = provider.get_chat_model(temperature=0.5)
        m2 = provider.get_chat_model(temperature=0.9)
        assert m1 is not m2

    def test_repr(self) -> None:
        cfg = ModelConfig(
            name="test", provider="ollama", model="test:1b",
            capabilities=[ModelCapability.GENERAL],
        )
        provider = OllamaProvider(cfg)
        r = repr(provider)
        assert "OllamaProvider" in r
        assert "test" in r

    def test_native_ollama_payload_attaches_base64_image_bytes(self, tmp_path: Path) -> None:
        """Vision payload uses Ollama's ``messages[].images``, never a path string."""
        image = tmp_path / "sample.jpg"
        Image.new("RGB", (20, 20), "white").save(image)
        cfg = ModelConfig(name="vision", provider="ollama", model="vision:latest",
                          capabilities=[ModelCapability.VISION])
        provider = OllamaProvider(cfg)
        encoded = provider.encode_images([str(image)])
        payload = provider._chat_payload(
            [{"role": "user", "content": "Describe the image."}],
            stream=True, image_paths=None, encoded_images=encoded,
        )
        assert payload["messages"][0]["images"] == encoded
        assert str(image) not in str(payload)
        assert encoded[0].startswith("/9j/")
        assert payload["think"] is False

    def test_invalid_image_is_rejected_before_request(self, tmp_path: Path) -> None:
        image = tmp_path / "not-an-image.jpg"
        image.write_text("not image bytes")
        cfg = ModelConfig(name="vision", provider="ollama", model="vision:latest",
                          capabilities=[ModelCapability.VISION])
        with pytest.raises(InvalidImageError):
            OllamaProvider(cfg).encode_images([str(image)])

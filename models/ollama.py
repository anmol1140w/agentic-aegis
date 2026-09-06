"""Ollama model provider — wraps langchain-ollama's ChatOllama.

This is the first (and currently only) concrete ModelProvider.
The rest of the application never imports ChatOllama directly.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from PIL import Image, UnidentifiedImageError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .base import ModelConfig, ModelProvider, ModelResponse
import os
from runtime.errors import EmptyGenerationError, InvalidImageError, ModelTimeoutError, ProviderError

_VISION_MAX_SIDE = int(os.environ.get("VISION_MAX_IMAGE_SIDE", "1024"))


class OllamaProvider(ModelProvider):
    """Model provider backed by a local Ollama instance."""

    def __init__(
        self,
        config: ModelConfig,
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(config, base_url)
        self._chat_model_cache: dict[str, ChatOllama] = {}

    # -- LangGraph integration ------------------------------------------

    def get_chat_model(self, **kwargs: Any) -> BaseChatModel:
        """Return a ``ChatOllama`` instance, cached per temperature."""
        temperature = kwargs.get("temperature", 0.7)
        cache_key = f"{self.config.model}:{temperature}"

        if cache_key not in self._chat_model_cache:
            self._chat_model_cache[cache_key] = ChatOllama(
                model=self.config.model,
                base_url=self.base_url,
                temperature=temperature,
                num_ctx=self.config.context_length,
                reasoning=True,
            )
        return self._chat_model_cache[cache_key]

    # -- Standalone generation ------------------------------------------

    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        return await self.chat([{"role": "user", "content": prompt}], **kwargs)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        payload = self._chat_payload(
            messages, stream=False, image_paths=kwargs.get("image_paths"),
            encoded_images=kwargs.get("encoded_images"),
            think=bool(kwargs.get("think", False)),
        )
        try:
            async with httpx.AsyncClient(timeout=kwargs.get("timeout", 120.0)) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError(f"Ollama timed out after {kwargs.get('timeout', 120.0)}s") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Ollama chat request failed: {exc}") from exc
        content = str(data.get("message", {}).get("content", "")).strip()
        if not content:
            raise EmptyGenerationError("Model returned no usable output")
        return ModelResponse(content=content, model=self.config.model, raw=data)

    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Call Ollama's native streaming API, including real base64 image bytes."""
        async for event in self.stream_chat_events(messages, **kwargs):
            if event.get("kind") == "content" and event.get("token"):
                yield str(event["token"])

    async def stream_chat_events(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream Ollama thinking/content events without buffering the response."""
        payload = self._chat_payload(
            messages, stream=True, image_paths=kwargs.get("image_paths"),
            encoded_images=kwargs.get("encoded_images"),
            think=bool(kwargs.get("think", False)),
        )
        timeout_seconds = float(kwargs.get("timeout", 120.0))
        received_output = False
        try:
            client_timeout = httpx.Timeout(timeout_seconds, connect=min(5.0, timeout_seconds))
            async with asyncio.timeout(timeout_seconds):
                async with httpx.AsyncClient(timeout=client_timeout) as client:
                    async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError as exc:
                                raise ProviderError("Ollama returned malformed streaming JSON") from exc
                            if event.get("error"):
                                raise ProviderError(f"Ollama error: {event['error']}")
                            message = event.get("message", {})
                            thinking = message.get("thinking", "")
                            token = message.get("content", "")
                            if thinking:
                                yield {"kind": "thinking", "token": str(thinking)}
                            if token:
                                received_output = True
                                yield {"kind": "content", "token": str(token)}
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelTimeoutError(f"Ollama stream exceeded {timeout_seconds:g}s deadline") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama streaming request failed: {exc}") from exc
        if not received_output:
            raise EmptyGenerationError("Model returned no usable output")

    def encode_images(self, paths: list[str]) -> list[str]:
        """Encode image files once so callers can time image loading separately."""
        return [_encode_image(path) for path in paths]

    def _chat_payload(
        self, messages: list[dict[str, Any]], *, stream: bool, image_paths: list[str] | None,
        encoded_images: list[str] | None = None, think: bool = False,
    ) -> dict[str, Any]:
        """Construct the documented Ollama ``/api/chat`` payload.

        Images belong on a message's ``images`` field as raw base64 data; a path
        in prompt text or an OpenAI-style ``image_url`` block is not sufficient.
        """
        payload_messages = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
        ]
        if image_paths or encoded_images:
            if not payload_messages:
                payload_messages.append({"role": "user", "content": ""})
            target = next((m for m in reversed(payload_messages) if m["role"] == "user"), payload_messages[-1])
            target["images"] = encoded_images if encoded_images is not None else [_encode_image(path) for path in image_paths or []]
        return {"model": self.config.model, "messages": payload_messages, "stream": stream,
                "think": think,
                "options": {"temperature": 0.7, "num_ctx": self.config.context_length}}

    # -- Health ---------------------------------------------------------

    async def health_check(self) -> bool:
        """Check Ollama is running and this model tag is pulled."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    return False
                data = resp.json()
                available_names = [
                    m.get("name", "") for m in data.get("models", [])
                ]
                return _model_available(self.config.model, available_names)
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
    "system": SystemMessage,
}


def _to_langchain_messages(messages: list[dict[str, str]]):
    """Convert ``[{"role": ..., "content": ...}]`` → LangChain messages."""
    out = []
    for msg in messages:
        cls = _ROLE_MAP.get(msg.get("role", "user"), HumanMessage)
        out.append(cls(content=msg.get("content", "")))
    return out


def _model_available(wanted: str, available: list[str]) -> bool:
    """Fuzzy match: ``qwen3:8b`` matches ``qwen3:8b``, ``qwen3:8b-fp16``, etc."""
    wanted_base = wanted.split(":")[0]
    for name in available:
        if name == wanted or name.startswith(wanted) or name.startswith(wanted_base + ":"):
            return True
    return False


def _encode_image(path: str | Path) -> str:
    """Validate and safely downscale an image before placing it in an Ollama request.

    Keeping very large engineering drawings at source resolution can exhaust the
    local Vulkan vision runner. OCR retains the original later; vision receives
    an in-memory 2048px rendition, never merely a file path.
    """
    image = Path(path)
    if not image.exists() or not image.is_file():
        raise InvalidImageError(f"Image file not found: {image}")
    try:
        data = image.read_bytes()
    except OSError as exc:
        raise InvalidImageError(f"Image cannot be read: {image}") from exc
    if len(data) < 12 or not _looks_like_image(data, image.suffix):
        raise InvalidImageError(f"Invalid or unsupported image: {image}")
    if image.suffix.lower() == ".svg":
        return base64.b64encode(data).decode("ascii")
    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            if max(source.size) <= _VISION_MAX_SIDE:
                encoded = data
            else:
                source.thumbnail((_VISION_MAX_SIDE, _VISION_MAX_SIDE))
                if source.mode not in ("RGB", "L"):
                    source = source.convert("RGB")
                buffer = io.BytesIO()
                source.save(buffer, format="JPEG", quality=90, optimize=True)
                encoded = buffer.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Invalid or unsupported image: {image}") from exc
    return base64.b64encode(encoded).decode("ascii")


def _looks_like_image(data: bytes, suffix: str) -> bool:
    signatures = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*")
    if data.startswith(signatures):
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return suffix.lower() == ".svg" and b"<svg" in data[:1024].lower()

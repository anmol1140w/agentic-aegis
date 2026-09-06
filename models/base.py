"""Abstract model provider interface and shared data types.

Every model backend (Ollama, vLLM, llama.cpp) implements ModelProvider.
The rest of the application depends only on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Capability enum — must match values used in config/models.yaml
# ---------------------------------------------------------------------------

class ModelCapability(str, Enum):
    """Capabilities a model may support."""

    GENERAL = "general"
    REASONING = "reasoning"
    CODING = "coding"
    DEBUGGING = "debugging"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    IMAGE_ANALYSIS = "image_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    SUMMARIZATION = "summarization"
    CALCULATION = "calculation"
    LIGHTWEIGHT = "lightweight"
    ENGINEERING_DOCUMENT = "engineering_document"
    ARTIFACT_GENERATION = "artifact_generation"


# ---------------------------------------------------------------------------
# Model configuration (loaded from YAML)
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Configuration for a single registered model."""

    name: str
    provider: str                               # e.g. "ollama"
    model: str                                  # provider-specific id, e.g. "qwen3:8b"
    capabilities: list[ModelCapability]
    context_length: int = 8192
    priority: int = 5                           # higher = preferred when tied
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Standardised response
# ---------------------------------------------------------------------------

class ModelResponse(BaseModel):
    """Standardised response returned by every provider."""

    content: str
    thinking: str | None = None
    model: str
    done: bool = True
    total_duration_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class ModelProvider(ABC):
    """Abstract interface for a model provider.

    Concrete implementations: OllamaProvider, (future) VLLMProvider, etc.
    """

    def __init__(self, config: ModelConfig, base_url: str = "http://localhost:11434"):
        self.config = config
        self.base_url = base_url

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model_id(self) -> str:
        return self.config.model

    # -- LangGraph integration ------------------------------------------

    @abstractmethod
    def get_chat_model(self, **kwargs: Any) -> BaseChatModel:
        """Return the underlying LangChain chat model for LangGraph nodes."""
        ...

    # -- Standalone generation (used outside LangGraph) -----------------

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> ModelResponse:
        """Single-turn text completion."""
        ...

    @abstractmethod
    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        """Multi-turn chat completion."""
        ...

    @abstractmethod
    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream visible response tokens from the provider."""
        ...

    # -- Health ---------------------------------------------------------

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable and this model is loaded."""
        ...

    # -- Repr -----------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} model={self.model_id!r}>"

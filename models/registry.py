"""Model registry — loads config/models.yaml, instantiates providers.

The registry is the single source of truth for which models are
configured.  Other layers (router, orchestrator) query it by
capability, name, or fallback status.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from .base import ModelCapability, ModelConfig, ModelProvider
from .ollama import OllamaProvider

if TYPE_CHECKING:
    pass

# Provider class lookup — extend when adding vLLM, llama.cpp, etc.
_PROVIDER_CLASSES: dict[str, type[ModelProvider]] = {
    "ollama": OllamaProvider,
}


class ModelRegistry:
    """Configuration‑driven model registry.

    Usage::

        registry = ModelRegistry.from_yaml("config/models.yaml")
        coder = registry.get_provider("qwen-coder")
    """

    def __init__(
        self,
        configs: dict[str, ModelConfig],
        base_url: str = "http://localhost:11434",
    ):
        self._configs = configs
        self._base_url = base_url
        self._providers: dict[str, ModelProvider] = {}
        self._build_providers()

    # -- Construction ---------------------------------------------------

    def _build_providers(self) -> None:
        for name, cfg in self._configs.items():
            cls = _PROVIDER_CLASSES.get(cfg.provider)
            if cls is None:
                raise ValueError(
                    f"Unknown provider '{cfg.provider}' for model '{name}'. "
                    f"Available: {list(_PROVIDER_CLASSES)}"
                )
            self._providers[name] = cls(cfg, self._base_url)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        base_url: str = "http://localhost:11434",
    ) -> ModelRegistry:
        """Load the registry from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")

        with open(path) as fh:
            raw = yaml.safe_load(fh)

        configs: dict[str, ModelConfig] = {}
        for name, entry in raw.get("models", {}).items():
            caps = [ModelCapability(c) for c in entry.get("capabilities", [])]
            configs[name] = ModelConfig(
                name=name,
                provider=entry["provider"],
                model=entry["model"],
                capabilities=caps,
                context_length=entry.get("context_length", 8192),
                priority=entry.get("priority", 5),
                is_fallback=entry.get("is_fallback", False),
            )
        return cls(configs, base_url)

    # -- Queries --------------------------------------------------------

    def get_provider(self, name: str) -> ModelProvider:
        """Get a provider by its registry name (e.g. ``'qwen-coder'``)."""
        if name not in self._providers:
            raise KeyError(
                f"Model '{name}' not registered. "
                f"Available: {list(self._providers)}"
            )
        return self._providers[name]

    def find_by_capability(
        self,
        *capabilities: ModelCapability,
    ) -> list[ModelProvider]:
        """Return providers that have **all** requested capabilities."""
        return [
            p
            for p in self._providers.values()
            if all(c in p.config.capabilities for c in capabilities)
        ]

    def get_fallback(self) -> ModelProvider | None:
        """Return the first provider marked ``is_fallback``."""
        for p in self._providers.values():
            if p.config.is_fallback:
                return p
        return None

    def list_models(self) -> list[ModelConfig]:
        """Return all registered model configs."""
        return list(self._configs.values())

    async def check_availability(self) -> dict[str, bool]:
        """Health‑check every registered model and return name → status."""
        results: dict[str, bool] = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.health_check()
            except Exception:
                results[name] = False
        return results

    @property
    def provider_names(self) -> list[str]:
        return list(self._providers.keys())

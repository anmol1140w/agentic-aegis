"""CPU-first operating profiles for local models."""
from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    context_chars: int = 48_000
    max_tool_calls: int = 40
    max_mutations: int = 10
    max_repairs: int = 3
    reviewer_model: str = "qwen-general"
    local_only: bool = True


def get_model_profile(model_id: str | None = None) -> ModelProfile:
    model = model_id or "qwen2.5-coder:7b"
    return ModelProfile(
        name="aegis-5b-cpu", model_id=model,
        context_chars=int(os.getenv("AEGIS_CONTEXT_CHARS", "48000")),
        max_tool_calls=int(os.getenv("AEGIS_MAX_TOOL_CALLS", "40")),
        max_mutations=int(os.getenv("AEGIS_MAX_MUTATIONS", "10")),
        max_repairs=int(os.getenv("AEGIS_MAX_REPAIRS", "3")),
        reviewer_model=os.getenv("AEGIS_REVIEWER_MODEL", "qwen-general"),
    )

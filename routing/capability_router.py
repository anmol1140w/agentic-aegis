"""Capability-first model routing with a quality floor and cost preference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models.base import ModelCapability, ModelConfig


@dataclass(frozen=True)
class CapabilityTask:
    task: str
    capabilities: tuple[ModelCapability, ...]
    quality_required: float = 0.80


@dataclass(frozen=True)
class Candidate:
    model: str
    matched: tuple[str, ...]
    quality: float
    cost: float


def choose_model(task: CapabilityTask, models: Iterable[ModelConfig]) -> Candidate:
    candidates: list[Candidate] = []
    required = set(task.capabilities)
    for model in models:
        matched = tuple(sorted(cap.value for cap in required.intersection(model.capabilities)))
        quality = len(matched) / max(1, len(required))
        if quality < task.quality_required:
            continue
        # Lower priority is treated as lower cost only as a deterministic
        # prototype heuristic; production configs can replace this with price.
        cost = max(0.1, 10.0 - model.priority)
        candidates.append(Candidate(model.name, matched, quality, cost))
    if not candidates:
        raise LookupError("No model clears the requested capability quality bar")
    return min(candidates, key=lambda item: (-item.quality, item.cost))

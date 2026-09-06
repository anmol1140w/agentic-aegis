"""Bounded capability discovery using cosine-ranked semantic representations."""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    left, right = list(a), list(b)
    if len(left) != len(right) or not left or not any(left) or not any(right):
        return 0.0
    denom = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / denom if denom else 0.0


@dataclass(frozen=True)
class CapabilityProfile:
    name: str
    description: str
    agent: str
    modality: str = "text"
    outputs: tuple[str, ...] = ()


class CapabilityMatcher:
    def __init__(self, profiles: list[CapabilityProfile] | None = None,
                 embedder: Callable[[str], list[float]] | None = None,
                 dimensions: int = 256):
        self.profiles = profiles or []
        self.embedder = embedder
        self.dimensions = dimensions
        self._cache: dict[str, tuple[str, list[float]]] = {}

    def _vector(self, text: str) -> list[float]:
        if self.embedder:
            return list(self.embedder(text))
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
            vector[index] += 1.0
        return vector

    def _profile_vector(self, profile: CapabilityProfile) -> list[float]:
        key = hashlib.sha256((profile.name + profile.description + repr(profile.outputs)).encode()).hexdigest()
        cached = self._cache.get(profile.name)
        if not cached or cached[0] != key:
            self._cache[profile.name] = (key, self._vector(f"{profile.name} {profile.description} {' '.join(profile.outputs)}"))
        return self._cache[profile.name][1]

    def rank(self, query: str, *, modality: str | None = None, output: str | None = None,
             top_k: int = 3) -> tuple[list[dict[str, Any]], dict[str, float]]:
        started = time.perf_counter()
        query_vector = self._vector(query)
        ranked = []
        for profile in self.profiles:
            score = cosine_similarity(query_vector, self._profile_vector(profile))
            if modality and profile.modality not in {modality, "text"}:
                continue
            if output and profile.outputs and output.lower() not in {x.lower() for x in profile.outputs}:
                continue
            ranked.append({"capability": profile.name, "agent": profile.agent, "score": round(score, 6)})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        metrics = {"embedding_ms": round((time.perf_counter() - started) * 1000, 3),
                   "similarity_ms": 0.0, "cache_size": len(self._cache)}
        return ranked[:max(1, top_k)], metrics


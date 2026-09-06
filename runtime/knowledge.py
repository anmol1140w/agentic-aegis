"""RAG / knowledge provider — stub interface.

RAG is explicitly deferred.  This module defines the interface so
future vector-DB / embedding / retrieval implementations can drop in
without changing the rest of the system.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class KnowledgeResult(BaseModel):
    """A single retrieval result."""

    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = {}


class KnowledgeProvider(ABC):
    """Abstract interface for knowledge retrieval (RAG)."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[KnowledgeResult]:
        ...


class NoOpKnowledgeProvider(KnowledgeProvider):
    """Placeholder — RAG is not implemented yet.

    Returns an empty list so callers degrade gracefully rather than
    crashing.
    """

    async def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[KnowledgeResult]:
        return []   # RAG deferred — see architecture docs

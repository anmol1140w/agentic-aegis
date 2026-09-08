"""Small dependency-free hybrid lexical/vector-like retrieval store.

The lexical index is deterministic and works offline.  An optional embedding
callback can add semantic scores without changing callers.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from runtime.knowledge import KnowledgeResult

_TOKEN = re.compile(r"[\w-]+", re.UNICODE)


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: Any = None


class HybridKnowledgeStore:
    def __init__(self, embed: Callable[[str], Any] | None = None):
        self.embed = embed
        self.chunks: dict[str, Chunk] = {}
        self._df: Counter[str] = Counter()

    def add(self, content: str, source: str, *, metadata: dict[str, Any] | None = None, chunk_size: int = 1200) -> list[str]:
        words = content.split()
        ids: list[str] = []
        for start in range(0, max(1, len(words)), chunk_size):
            text = " ".join(words[start:start + chunk_size])
            if not text:
                continue
            cid = uuid.uuid4().hex
            chunk = Chunk(cid, text, source, metadata or {}, self.embed(text) if self.embed else None)
            self.chunks[cid] = chunk
            self._df.update(set(self._tokens(text)))
            ids.append(cid)
        return ids

    async def search(self, query: str, *, top_k: int = 5, alpha: float = 0.65, **_: Any) -> list[KnowledgeResult]:
        query_tokens = self._tokens(query)
        if not query_tokens:
            return []
        query_vector = self.embed(query) if self.embed else None
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks.values():
            tokens = self._tokens(chunk.content)
            tf = Counter(tokens)
            lexical = sum((tf[token] / max(1, len(tokens))) * math.log(1 + len(self.chunks) / (1 + self._df[token])) for token in query_tokens)
            semantic = self._semantic(query_vector, chunk.vector) if query_vector is not None and chunk.vector is not None else 0.0
            score = alpha * lexical + (1 - alpha) * semantic
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [KnowledgeResult(content=c.content, source=c.source, score=score, metadata={"chunk_id": c.id, **c.metadata}) for score, c in scored[:top_k]]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN.findall(text)]

    @staticmethod
    def _semantic(query: Any, vector: Any) -> float:
        if not isinstance(query, (list, tuple)) or not isinstance(vector, (list, tuple)) or len(query) != len(vector):
            return 0.0
        qnorm = math.sqrt(sum(float(x) ** 2 for x in query))
        vnorm = math.sqrt(sum(float(x) ** 2 for x in vector))
        return sum(float(a) * float(b) for a, b in zip(query, vector)) / (qnorm * vnorm) if qnorm and vnorm else 0.0

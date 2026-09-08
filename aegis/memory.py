"""Durable, scoped agent memory with explicit retention and search."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class MemoryStore:
    def __init__(self, path: str | Path = ".aegis/memory.sqlite3") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, scope TEXT, text TEXT, metadata TEXT, created REAL, expires REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def put(self, text: str, *, scope: str = "global", metadata: dict[str, Any] | None = None, ttl_seconds: float | None = None) -> int:
        expires = time.time() + ttl_seconds if ttl_seconds else None
        with self._connect() as db:
            cur = db.execute("INSERT INTO memories(scope,text,metadata,created,expires) VALUES(?,?,?,?,?)", (scope, text, json.dumps(metadata or {}), time.time(), expires))
            return int(cur.lastrowid)

    def search(self, query: str, *, scope: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        words = {word.lower() for word in query.split() if word}
        now = time.time()
        sql = "SELECT * FROM memories WHERE (expires IS NULL OR expires > ?)"
        params: list[Any] = [now]
        if scope:
            sql += " AND scope = ?"
            params.append(scope)
        rows = self._connect().execute(sql, params).fetchall()
        ranked = []
        for row in rows:
            score = sum(word in row["text"].lower() for word in words)
            if score:
                ranked.append((score, {"id": row["id"], "scope": row["scope"], "text": row["text"], "metadata": json.loads(row["metadata"])}))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:limit]]

    def forget(self, *, scope: str | None = None, memory_id: int | None = None) -> int:
        if scope is None and memory_id is None:
            raise ValueError("scope or memory_id is required")
        with self._connect() as db:
            if memory_id is not None:
                cur = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                cur = db.execute("DELETE FROM memories WHERE scope = ?", (scope,))
            return cur.rowcount

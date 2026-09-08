"""Durable, scoped agent memory with explicit retention and search."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from enum import StrEnum


class MemoryKind(StrEnum):
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    RELATIONAL = "relational"


class MemoryStore:
    def __init__(self, path: str | Path = ".aegis/memory.sqlite3") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, scope TEXT, text TEXT, metadata TEXT, created REAL, expires REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)")
            columns = {row[1] for row in db.execute("PRAGMA table_info(memories)").fetchall()}
            if "kind" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN kind TEXT NOT NULL DEFAULT 'semantic'")
            if "importance" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN importance REAL NOT NULL DEFAULT 0.5")
            if "provenance" not in columns:
                db.execute("ALTER TABLE memories ADD COLUMN provenance TEXT NOT NULL DEFAULT ''")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def put(self, text: str, *, scope: str = "global", metadata: dict[str, Any] | None = None,
            ttl_seconds: float | None = None, kind: MemoryKind | str = MemoryKind.SEMANTIC,
            importance: float = 0.5, provenance: str = "") -> int:
        if not str(text).strip():
            raise ValueError("memory text is required")
        importance = max(0.0, min(1.0, float(importance)))
        expires = time.time() + ttl_seconds if ttl_seconds else None
        with self._connect() as db:
            cur = db.execute(
                "INSERT INTO memories(scope,text,metadata,created,expires,kind,importance,provenance) VALUES(?,?,?,?,?,?,?,?)",
                (scope, text, json.dumps(metadata or {}), time.time(), expires, str(kind), importance, provenance),
            )
            return int(cur.lastrowid)

    def record_event(self, event: str, *, scope: str = "global", metadata: dict[str, Any] | None = None) -> int:
        """Persist an episodic observation suitable for replay and audit."""
        return self.put(event, scope=scope, metadata=metadata, kind=MemoryKind.EPISODIC,
                        importance=float((metadata or {}).get("importance", 0.5)), provenance="runtime")

    def remember(self, text: str, *, scope: str = "global", kind: MemoryKind | str = MemoryKind.SEMANTIC,
                 metadata: dict[str, Any] | None = None, ttl_seconds: float | None = None,
                 importance: float = 0.5, provenance: str = "user") -> int:
        """Explicit semantic/relational memory API; aliases are intentional."""
        return self.put(text, scope=scope, kind=kind, metadata=metadata, ttl_seconds=ttl_seconds,
                        importance=importance, provenance=provenance)

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
                ranked.append((score + float(row["importance"] or 0.0) * 0.01, {
                    "id": row["id"], "scope": row["scope"], "text": row["text"],
                    "kind": row["kind"], "importance": row["importance"],
                    "provenance": row["provenance"], "metadata": json.loads(row["metadata"]),
                }))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in ranked[:limit]]

    def prune(self, *, scope: str | None = None, before: float | None = None,
              keep_importance_at_least: float | None = None) -> int:
        """Apply explicit retention policy; never silently delete memories."""
        clauses = ["expires IS NOT NULL AND expires <= ?"]
        params: list[Any] = [time.time()]
        if scope:
            clauses.append("scope = ?"); params.append(scope)
        if before is not None:
            clauses.append("created < ?"); params.append(before)
        if keep_importance_at_least is not None:
            clauses.append("importance < ?"); params.append(keep_importance_at_least)
        with self._connect() as db:
            cur = db.execute("DELETE FROM memories WHERE " + " AND ".join(clauses), params)
            return cur.rowcount

    def forget(self, *, scope: str | None = None, memory_id: int | None = None) -> int:
        if scope is None and memory_id is None:
            raise ValueError("scope or memory_id is required")
        with self._connect() as db:
            if memory_id is not None:
                cur = db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            else:
                cur = db.execute("DELETE FROM memories WHERE scope = ?", (scope,))
            return cur.rowcount

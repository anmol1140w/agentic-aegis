"""Persistent task state — SQLite-backed.

Stores task lifecycle: creation, steps, model used, artifacts, errors.
Abstract interface so PostgreSQL can replace SQLite later.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TaskRecord:
    """In-memory representation of a persisted task."""

    __slots__ = (
        "task_id", "status", "current_step", "task_type", "modality",
        "model_used", "messages_json", "tool_calls_json",
        "artifacts_json", "errors_json", "created_at", "updated_at",
    )

    def __init__(self, **kwargs: Any):
        for slot in self.__slots__:
            setattr(self, slot, kwargs.get(slot))

    def to_dict(self) -> dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


class BaseTaskStore(ABC):
    """Abstract task store interface."""

    @abstractmethod
    def save(self, record: TaskRecord) -> None: ...

    @abstractmethod
    def get(self, task_id: str) -> TaskRecord | None: ...

    @abstractmethod
    def update_status(self, task_id: str, status: str, step: str) -> None: ...

    @abstractmethod
    def list_recent(self, limit: int = 20) -> list[TaskRecord]: ...


class SQLiteTaskStore(BaseTaskStore):
    """SQLite-backed task store.

    Uses synchronous sqlite3 (sufficient for single-user workstation).
    """

    def __init__(self, db_path: str | Path = "./data/workbench.db"):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id       TEXT PRIMARY KEY,
                status        TEXT NOT NULL DEFAULT 'created',
                current_step  TEXT,
                task_type     TEXT,
                modality      TEXT,
                model_used    TEXT,
                messages_json TEXT DEFAULT '[]',
                tool_calls_json TEXT DEFAULT '[]',
                artifacts_json TEXT DEFAULT '[]',
                errors_json   TEXT DEFAULT '[]',
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def save(self, record: TaskRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tasks
                (task_id, status, current_step, task_type, modality,
                 model_used, messages_json, tool_calls_json,
                 artifacts_json, errors_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.task_id,
                record.status or "created",
                record.current_step,
                record.task_type,
                record.modality,
                record.model_used,
                record.messages_json or "[]",
                record.tool_calls_json or "[]",
                record.artifacts_json or "[]",
                record.errors_json or "[]",
                record.created_at or now,
                now,
            ),
        )
        self._conn.commit()

    def get(self, task_id: str) -> TaskRecord | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            return None
        return TaskRecord(**dict(row))

    def update_status(self, task_id: str, status: str, step: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE tasks SET status = ?, current_step = ?, updated_at = ? WHERE task_id = ?",
            (status, step, now, task_id),
        )
        self._conn.commit()

    def list_recent(self, limit: int = 20) -> list[TaskRecord]:
        rows = self._conn.execute(
            "SELECT * FROM tasks ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [TaskRecord(**dict(r)) for r in rows]

    def close(self) -> None:
        self._conn.close()

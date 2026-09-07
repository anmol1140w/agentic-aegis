"""Durable SQLite snapshots for resumable graph execution."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteGraphStateStore:
    def __init__(self, db_path: str | Path = "./data/workbench.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self._init_schema()
        self.conn.commit()

    def _init_schema(self) -> None:
        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(graph_checkpoints)")]
        if columns and "snapshot_id" not in columns:
            self.conn.execute("ALTER TABLE graph_checkpoints RENAME TO graph_checkpoints_legacy")
            columns = []
        self.conn.execute("CREATE TABLE IF NOT EXISTS graph_checkpoints (snapshot_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, task_id TEXT, node TEXT NOT NULL, state_json TEXT NOT NULL, updated_at TEXT NOT NULL)")
        if not columns:
            legacy = self.conn.execute("SELECT run_id, task_id, node, state_json, updated_at FROM graph_checkpoints_legacy").fetchall() if self._table_exists("graph_checkpoints_legacy") else []
            for run_id, task_id, node, state_json, updated_at in legacy:
                self.conn.execute("INSERT OR IGNORE INTO graph_checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                                   (f"{run_id}:{node}:legacy", run_id, task_id, node, state_json, updated_at))
            self.conn.execute("DROP TABLE IF EXISTS graph_checkpoints_legacy")

    def _table_exists(self, name: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

    def save(self, run_id: str, task_id: str, node: str, state: dict[str, Any]) -> None:
        updated = datetime.now(timezone.utc).isoformat()
        snapshot_id = f"{run_id}:{node}:{updated}"
        self.conn.execute("INSERT INTO graph_checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                          (snapshot_id, run_id, task_id, node, json.dumps(state, default=str), updated))
        self.conn.commit()

    def load(self, run_id: str) -> dict[str, Any] | None:
        item = self.load_latest(run_id)
        return item[1] if item else None

    def load_latest(self, run_id: str) -> tuple[str, dict[str, Any]] | None:
        row = self.conn.execute("SELECT node, state_json FROM graph_checkpoints WHERE run_id = ? ORDER BY updated_at DESC LIMIT 1", (run_id,)).fetchone()
        return (row[0], json.loads(row[1])) if row else None

    def latest(self, task_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT state_json FROM graph_checkpoints WHERE task_id = ? ORDER BY updated_at DESC LIMIT 1", (task_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self.conn.close()

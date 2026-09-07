"""Structured execution telemetry stored locally in SQLite."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class SQLiteTelemetryStore:
    def __init__(self, db_path: str | Path = "./data/workbench.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute("""CREATE TABLE IF NOT EXISTS execution_telemetry (
            run_id TEXT PRIMARY KEY, task_type TEXT, required_capabilities_json TEXT,
            selected_model TEXT, alternative_models_json TEXT, latency_ms REAL,
            token_count INTEGER, tool_calls INTEGER, tool_failures INTEGER,
            verification_result INTEGER, human_approval INTEGER, final_success INTEGER,
            metadata_json TEXT, routing_context_json TEXT, eligible_candidates_json TEXT,
            selected_workflow TEXT, routing_mode TEXT, adaptive_fallback INTEGER,
            escalation INTEGER, reward REAL, security_violations INTEGER,
            unauthorized_tool_executions INTEGER, approval_bypasses INTEGER,
            created_at REAL NOT NULL)""")
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(execution_telemetry)")}
        additions = {
            "routing_context_json": "TEXT", "eligible_candidates_json": "TEXT",
            "selected_workflow": "TEXT", "routing_mode": "TEXT DEFAULT 'deterministic'",
            "adaptive_fallback": "INTEGER DEFAULT 0", "escalation": "INTEGER DEFAULT 0",
            "reward": "REAL DEFAULT 0", "security_violations": "INTEGER DEFAULT 0",
            "unauthorized_tool_executions": "INTEGER DEFAULT 0", "approval_bypasses": "INTEGER DEFAULT 0",
        }
        for name, kind in additions.items():
            if name not in columns:
                self.conn.execute(f"ALTER TABLE execution_telemetry ADD COLUMN {name} {kind}")
        self.conn.commit()

    def record(self, *, run_id: str, task_type: str, required_capabilities: list[str],
               selected_model: str, alternative_models: list[str] | None = None,
               latency_ms: float = 0.0, token_count: int = 0, tool_calls: int = 0,
               tool_failures: int = 0, verification_result: bool = False,
               human_approval: bool = False, final_success: bool = False,
               metadata: dict[str, Any] | None = None,
               routing_context: dict[str, Any] | None = None,
               eligible_candidates: list[dict[str, Any]] | None = None,
               selected_workflow: str = "", routing_mode: str = "deterministic",
               adaptive_fallback: bool = False, escalation: bool = False,
               reward: float = 0.0, security_violations: int = 0,
               unauthorized_tool_executions: int = 0, approval_bypasses: int = 0) -> dict[str, Any]:
        record = {"run_id": run_id, "task_type": task_type,
                  "required_capabilities": required_capabilities,
                  "selected_model": selected_model,
                  "alternative_models": alternative_models or [],
                  "latency_ms": round(latency_ms, 2), "token_count": token_count,
                  "tool_calls": tool_calls, "tool_failures": tool_failures,
                  "verification_result": bool(verification_result),
                  "human_approval": bool(human_approval),
                  "final_success": bool(final_success), "metadata": metadata or {},
                  "routing_context": routing_context or {},
                  "eligible_candidates": eligible_candidates or [],
                  "selected_workflow": selected_workflow, "routing_mode": routing_mode,
                  "adaptive_fallback": bool(adaptive_fallback), "escalation": bool(escalation),
                  "reward": float(reward), "security_violations": int(security_violations),
                  "unauthorized_tool_executions": int(unauthorized_tool_executions),
                  "approval_bypasses": int(approval_bypasses),
                  "created_at": time.time()}
        fields = ["run_id", "task_type", "required_capabilities_json", "selected_model", "alternative_models_json",
                  "latency_ms", "token_count", "tool_calls", "tool_failures", "verification_result", "human_approval",
                  "final_success", "metadata_json", "routing_context_json", "eligible_candidates_json", "selected_workflow",
                  "routing_mode", "adaptive_fallback", "escalation", "reward", "security_violations",
                  "unauthorized_tool_executions", "approval_bypasses", "created_at"]
        values = [run_id, task_type, json.dumps(required_capabilities), selected_model,
                  json.dumps(alternative_models or []), record["latency_ms"], token_count, tool_calls, tool_failures,
                  int(verification_result), int(human_approval), int(final_success), json.dumps(metadata or {}, default=str),
                  json.dumps(routing_context or {}, default=str), json.dumps(eligible_candidates or [], default=str),
                  selected_workflow, routing_mode, int(adaptive_fallback), int(escalation), record["reward"],
                  record["security_violations"], record["unauthorized_tool_executions"], record["approval_bypasses"],
                  record["created_at"]]
        self.conn.execute(f"INSERT OR REPLACE INTO execution_telemetry ({', '.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)
        self.conn.commit()
        return record

    def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM execution_telemetry ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        columns = [item[0] for item in cursor.description]
        result = []
        for row in rows:
            item = dict(zip(columns, row))
            for field in ("required_capabilities_json", "alternative_models_json", "metadata_json",
                          "routing_context_json", "eligible_candidates_json"):
                item[field.removesuffix("_json")] = json.loads(item.pop(field) or "{}")
            for field in ("verification_result", "human_approval", "final_success", "adaptive_fallback", "escalation"):
                item[field] = bool(item[field])
            result.append(item)
        return result

    def close(self) -> None:
        self.conn.close()

"""Structured audit logging.

Append-only JSONL log capturing every significant operation:
task creation, model invocation, tool execution, approvals, errors.

Does NOT log raw document content by default to protect sensitive data.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger("sih.audit")


class AuditEntry(BaseModel):
    """A single audit record."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    event: str                           # e.g. "task_created", "model_invoked"
    task_id: str | None = None
    user: str = "local"
    agent: str | None = None
    model: str | None = None
    tool: str | None = None
    action: str | None = None
    status: str | None = None            # success / failure / partial
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class AuditLogger:
    """Append-only JSONL audit logger.

    Usage::

        audit = AuditLogger("./logs/audit.jsonl")
        audit.log("model_invoked", task_id="abc", model="qwen3:8b")
    """

    def __init__(self, path: str | Path = "./logs/audit.jsonl"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")

    def log(
        self,
        event: str,
        *,
        task_id: str | None = None,
        model: str | None = None,
        tool: str | None = None,
        action: str | None = None,
        status: str | None = None,
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            event=event,
            task_id=task_id,
            model=model,
            tool=tool,
            action=action,
            status=status,
            duration_ms=duration_ms,
            metadata=metadata or {},
            error=error,
        )
        line = entry.model_dump_json() + "\n"
        self._fh.write(line)
        self._fh.flush()
        logger.debug("AUDIT: %s", line.rstrip())
        return entry

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> AuditLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

"""Enterprise governance: hierarchical RBAC, policy evaluation and audit chain."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Principal:
    subject: str
    roles: frozenset[str] = frozenset({"user"})
    groups: frozenset[str] = frozenset()
    department: str = ""


@dataclass
class RBAC:
    role_permissions: dict[str, set[str]] = field(default_factory=lambda: {
        "user": {"task:read", "task:create", "rag:read"},
        "operator": {"task:read", "task:create", "rag:read", "tool:use", "workflow:run"},
        "reviewer": {"task:read", "task:create", "rag:read", "workflow:approve"},
        "admin": {"*"},
    })
    group_permissions: dict[str, set[str]] = field(default_factory=dict)

    def allows(self, principal: Principal, permission: str) -> bool:
        grants = set().union(*(self.role_permissions.get(role, set()) for role in principal.roles))
        grants |= set().union(*(self.group_permissions.get(group, set()) for group in principal.groups))
        return "*" in grants or permission in grants

    def require(self, principal: Principal, permission: str) -> None:
        if not self.allows(principal, permission):
            raise PermissionError(f"{principal.subject} lacks permission {permission}")


@dataclass
class PolicyEngine:
    rbac: RBAC = field(default_factory=RBAC)
    department_models: dict[str, set[str]] = field(default_factory=dict)
    department_tools: dict[str, set[str]] = field(default_factory=dict)
    restricted_groups: set[str] = field(default_factory=set)

    def check(self, principal: Principal, action: str, *, model: str | None = None, tool: str | None = None) -> None:
        self.rbac.require(principal, action)
        if principal.groups & self.restricted_groups and action in {"tool:use", "workflow:run"}:
            raise PermissionError("restricted group requires explicit approval for this action")
        if model and principal.department in self.department_models and model not in self.department_models[principal.department]:
            raise PermissionError(f"model {model} is not approved for department {principal.department}")
        if tool and principal.department in self.department_tools and tool not in self.department_tools[principal.department]:
            raise PermissionError(f"tool {tool} is not approved for department {principal.department}")


class AuditChain:
    """Append-only, hash-chained JSONL audit records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._previous = self._last_hash()

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "0" * 64
        try:
            last = self.path.read_text(encoding="utf-8").splitlines()[-1]
            return json.loads(last).get("hash", "0" * 64)
        except (OSError, IndexError, json.JSONDecodeError):
            return "0" * 64

    def append(self, event: str, *, actor: str = "local", **data: Any) -> dict[str, Any]:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, "actor": actor, **data}
        with self._lock:
            payload = json.dumps({"previous": self._previous, **record}, sort_keys=True, default=str)
            digest = hashlib.sha256(payload.encode()).hexdigest()
            record = {"previous": self._previous, **record, "hash": digest}
            self.path.open("a", encoding="utf-8").write(json.dumps(record, sort_keys=True, default=str) + "\n")
            self._previous = digest
        return record

    def verify(self) -> bool:
        previous = "0" * 64
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                digest = record.pop("hash")
                record_previous = record.get("previous")
                if record_previous != previous:
                    return False
                if hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest() != digest:
                    return False
                previous = digest
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

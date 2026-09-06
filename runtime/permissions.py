"""Session-scoped permission requests and saved rules for agent actions."""

from __future__ import annotations

import fnmatch
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PermissionEffect(StrEnum):
    ALLOW = "allow"
    PROMPT = "prompt"
    DENY = "deny"


@dataclass(frozen=True)
class PermissionRule:
    action: str
    resource: str = "*"
    effect: PermissionEffect = PermissionEffect.PROMPT
    session_id: str | None = None


@dataclass
class PermissionRequest:
    request_id: str
    session_id: str
    action: str
    resources: tuple[str, ...]
    agent: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    message: str = ""


class PermissionManager:
    """Deterministic permission service with deny-over-prompt-over-allow precedence."""

    def __init__(self, rules: list[PermissionRule] | None = None) -> None:
        self._rules = list(rules or [])
        self._requests: dict[str, PermissionRequest] = {}

    def evaluate(self, session_id: str, action: str, resources: list[str] | tuple[str, ...]) -> PermissionEffect:
        matched = [rule for rule in self._rules if self._matches(rule, session_id, action, resources)]
        if any(rule.effect is PermissionEffect.DENY for rule in matched):
            return PermissionEffect.DENY
        if any(rule.effect is PermissionEffect.PROMPT for rule in matched):
            return PermissionEffect.PROMPT
        if matched:
            return PermissionEffect.ALLOW
        return PermissionEffect.PROMPT

    def request(self, session_id: str, action: str, resources: list[str] | tuple[str, ...], *, agent: str = "", metadata: dict[str, Any] | None = None) -> PermissionRequest:
        req = PermissionRequest(f"perm_{uuid.uuid4().hex[:12]}", session_id, action, tuple(resources), agent, metadata or {})
        effect = self.evaluate(session_id, action, resources)
        if effect is PermissionEffect.DENY:
            req.status = "denied"
            req.message = "Denied by permission rule."
        elif effect is PermissionEffect.ALLOW:
            req.status = "approved"
            req.message = "Allowed by permission rule."
        self._requests[req.request_id] = req
        return req

    def reply(self, request_id: str, approved: bool, *, message: str = "", save: bool = False) -> PermissionRequest:
        req = self._requests[request_id]
        if req.status != "pending":
            return req
        req.status = "approved" if approved else "denied"
        req.message = message[:500]
        if save and approved:
            self._rules.append(PermissionRule(req.action, req.resources[0] if req.resources else "*", PermissionEffect.ALLOW, req.session_id))
        return req

    def get(self, request_id: str) -> PermissionRequest | None:
        return self._requests.get(request_id)

    def list_pending(self, session_id: str | None = None) -> list[PermissionRequest]:
        return [r for r in self._requests.values() if r.status == "pending" and (session_id is None or r.session_id == session_id)]

    def rules(self, session_id: str | None = None) -> list[PermissionRule]:
        return [r for r in self._rules if session_id is None or r.session_id in {None, session_id}]

    @staticmethod
    def _matches(rule: PermissionRule, session_id: str, action: str, resources: list[str] | tuple[str, ...]) -> bool:
        if rule.session_id not in {None, session_id} or not fnmatch.fnmatchcase(action, rule.action):
            return False
        return all(fnmatch.fnmatchcase(resource, rule.resource) for resource in resources) if resources else True

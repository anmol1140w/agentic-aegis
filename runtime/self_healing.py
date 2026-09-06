"""Bounded, policy-neutral self-healing decisions."""
from __future__ import annotations

from typing import Any


NON_RETRYABLE = {"approval_denied", "OutsideWorkspace", "path_escape", "disallowed_command", "permission_denied"}


def classify_failure(result: dict[str, Any]) -> dict[str, Any]:
    code = str(result.get("error") or result.get("error_code") or "tool_failure")
    retryable = code not in NON_RETRYABLE and not result.get("policy_denied", False)
    return {"error_code": code, "retryable": retryable,
            "reason": "recoverable tool failure" if retryable else "policy or permission failure"}


def repair_budget_available(step: dict[str, Any], *, tool_calls: int, max_tool_calls: int) -> bool:
    return tool_calls < max_tool_calls and int(step.get("retry_count", 0)) < int(step.get("max_retries", 0))


def normalize_repair_action(action: dict[str, Any]) -> dict[str, Any]:
    if action.get("action") not in {"retry", "replace_step", "skip", "abort"}:
        raise ValueError("unsupported repair action")
    if action["action"] == "replace_step" and not isinstance(action.get("parameters", {}), dict):
        raise ValueError("repair parameters must be an object")
    return {"action": action["action"], "tool": action.get("tool"),
            "parameters": action.get("parameters", {}), "reason": str(action.get("reason", ""))[:500]}


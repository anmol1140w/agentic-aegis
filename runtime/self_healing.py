"""Bounded, policy-neutral self-healing decisions."""
from __future__ import annotations

from typing import Any

from runtime.contracts import FailureRecord


NON_RETRYABLE = {
    "approval_denied", "OutsideWorkspace", "path_escape", "disallowed_command",
    "permission_denied", "path_not_established", "edit-before-read blocked",
}


def classify_failure(result: dict[str, Any]) -> dict[str, Any]:
    code_value = result.get("error") or result.get("error_code")
    if not code_value and isinstance(result.get("errors"), list) and result["errors"]:
        first = result["errors"][0]
        code_value = first.get("code") if isinstance(first, dict) else first
    code = str(code_value or "tool_failure")
    retryable = code not in NON_RETRYABLE and not result.get("policy_denied", False)
    return {"error_code": code, "retryable": retryable,
            "reason": "recoverable tool failure" if retryable else "policy or permission failure"}


def failure_record(result: dict[str, Any], *, command: str = "", file: str = "") -> FailureRecord:
    """Normalize heterogeneous tool/test errors into the AEGIS taxonomy."""
    raw = str((result.get("error") or result.get("error_code") or
               (result.get("errors") or ["unknown"])[0])).lower()
    if "security" in raw or "injection" in raw:
        category = "SECURITY_BLOCK"
    elif "permission" in raw or "approval" in raw or "outsideworkspace" in raw or "path_escape" in raw:
        category = "PERMISSION_ERROR"
    elif "timeout" in raw:
        category = "TIMEOUT"
    elif "syntax" in raw:
        category = "SYNTAX_ERROR"
    elif "import" in raw or "module" in raw:
        category = "IMPORT_ERROR"
    elif "type" in raw:
        category = "TYPE_ERROR"
    elif "test" in raw or "assert" in raw:
        category = "TEST_FAILURE"
    elif "path" in raw or "file" in raw:
        category = "PATH_ERROR"
    elif "depend" in raw:
        category = "DEPENDENCY_ERROR"
    elif raw in {"tool_failure", "unknown", ""}:
        category = "UNKNOWN"
    else:
        category = "COMMAND_FAILURE"
    classified = classify_failure(result)
    return FailureRecord(category=category, code=classified["error_code"], command=command,
                         file=file, stdout_summary=str(result.get("stdout", ""))[-1000:],
                         stderr_summary=str(result.get("stderr", ""))[-1000:],
                         retryable=classified["retryable"],
                         repair_strategy="stop_and_request_approval" if not classified["retryable"] else "inspect_and_retry")


def repair_budget_available(step: dict[str, Any], *, tool_calls: int, max_tool_calls: int) -> bool:
    return tool_calls < max_tool_calls and int(step.get("retry_count", 0)) < int(step.get("max_retries", 0))


def normalize_repair_action(action: dict[str, Any]) -> dict[str, Any]:
    if action.get("action") not in {"retry", "replace_step", "skip", "abort"}:
        raise ValueError("unsupported repair action")
    if action["action"] == "replace_step" and not isinstance(action.get("parameters", {}), dict):
        raise ValueError("repair parameters must be an object")
    return {"action": action["action"], "tool": action.get("tool"),
            "parameters": action.get("parameters", {}), "reason": str(action.get("reason", ""))[:500]}

"""Deterministic evidence checks for tool and task results."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re


def verify_tool_result(result: dict[str, Any], *, workspace_root: str | Path | None = None) -> dict[str, Any]:
    """Verify execution evidence without trusting model prose."""
    ok = bool(result.get("ok", result.get("success", result.get("status") in {"success", "succeeded", "verified"})))
    evidence: list[dict[str, Any]] = []
    if "exit_code" in result:
        passed = result.get("exit_code") == 0 and not result.get("timed_out", False)
        evidence.append({"kind": "command", "passed": passed, "source": result.get("tool", "command"),
                         "detail": f"exit_code={result.get('exit_code')}"})
        ok = ok and passed
    path = result.get("path")
    if path and workspace_root:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(workspace_root) / candidate
        inside = False
        try:
            candidate.resolve().relative_to(Path(workspace_root).resolve())
            inside = True
        except ValueError:
            inside = False
        exists = candidate.exists() and candidate.is_file()
        evidence.append({"kind": "file", "passed": inside and exists, "source": "filesystem",
                         "path": str(candidate), "observed": {"inside_workspace": inside, "exists": exists}})
        ok = ok and inside and exists
    if result.get("timed_out"):
        ok = False
        evidence.append({"kind": "timeout", "passed": False, "source": result.get("tool", "tool")})
    result_ok = result.get("ok", result.get("success", result.get("status") in {"success", "succeeded", "verified"}))
    if not result_ok:
        evidence.append({"kind": "tool", "passed": False, "source": result.get("tool", "tool"),
                         "detail": str(result.get("error") or result.get("message") or "tool failed")[:500]})
    return {"passed": ok, "status": "verified" if ok else "failed", "summary":
            "deterministic evidence passed" if ok else "deterministic evidence failed", "evidence": evidence,
            "missing_evidence": [] if ok else ["successful tool evidence"]}


def verify_plan(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [verify_tool_result(item) for item in step_results]
    evidence = [entry for check in checks for entry in check.get("evidence", [])]
    passed = bool(checks) and all(check.get("passed", False) for check in checks)
    return {"passed": passed, "status": "verified" if passed else "failed",
            "summary": "all steps have deterministic evidence" if passed else "one or more steps lack valid evidence",
            "evidence": evidence, "missing_evidence": [] if passed else ["successful evidence for every step"]}


def coding_execution_requirements(task: str) -> dict[str, bool]:
    """Return deterministic execution requirements for coding work."""
    text = str(task).lower()
    coding = bool(re.search(
        r"\b(code|coding|source|test|tests|pytest|compile|compilation|debug|debugging|"
        r"fix|repair|implement|implementation|function|module|script)\b|\.(py|js|ts|rs|go|java|c|cpp)\b",
        text,
    ))
    test_task = bool(re.search(r"\b(pytest|test|tests|testing|test-suite|unittest|compile|compil(?:e|ation))\b", text))
    execution_task = bool(re.search(r"\b(run|execute|verify|build|compile|debug|debugging|fix|repair|implement)\b", text))
    return {"coding": coding, "requires_command": coding and (test_task or execution_task),
            "requires_test_command": test_task}


def verify_coding_result(result: dict[str, Any], task: str) -> dict[str, Any]:
    """Verify coding-agent success from recorded execution, never model prose."""
    requirements = coding_execution_requirements(task)
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    state = metadata.get("coding_state") if isinstance(metadata.get("coding_state"), dict) else {}
    commands = state.get("commands") if isinstance(state.get("commands"), list) else []
    successful = [item for item in commands if isinstance(item, dict) and item.get("exit_code") == 0 and not item.get("timed_out")]
    test_commands = [item for item in successful if re.search(r"\b(pytest|unittest|cargo\s+test|go\s+test|npm\s+test|make\s+test)\b", str(item.get("command", "")), re.I)]
    evidence = [{"kind": "execute_command", "passed": bool(successful), "source": "coding_state",
                 "detail": f"{len(successful)} successful command(s) recorded"}]
    missing: list[str] = []
    if requirements["requires_command"] and not commands:
        missing.append("actual execute_command evidence")
    if requirements["requires_test_command"] and not test_commands:
        missing.append("successful test command with exit_code=0")
    if requirements["requires_test_command"] and test_commands:
        last_edit = int(state.get("last_edit_state_version", -1) or -1)
        if last_edit >= 0 and not any(int(item.get("state_version", -1)) >= last_edit for item in test_commands):
            missing.append("test command executed after the resulting edit")
    if requirements["requires_command"] and state.get("verification", {}).get("status") not in {"passed", "verified"}:
        missing.append("deterministic verification state")
    if result.get("status") not in {"success", "succeeded"}:
        missing.append("successful coding specialist result")
    passed = not missing
    return {"passed": passed, "status": "verified" if passed else "failed",
            "summary": "coding execution evidence passed" if passed else "coding result lacks required execution evidence",
            "evidence": evidence, "missing_evidence": missing}

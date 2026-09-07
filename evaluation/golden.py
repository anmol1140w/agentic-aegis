"""Repeatable deterministic golden-task evaluation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from routing.classifier import TaskClassifier
from runtime.tool_policy import PolicyEngine, ToolPolicy


def load_golden_tasks(path: str | Path | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path else Path(__file__).with_name("golden_tasks.json")
    return json.loads(source.read_text(encoding="utf-8"))


def _matches(task: Any, expected: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for field in ("domain_intent", "workflow", "task_type", "execution_mode"):
        if field in expected:
            observed = getattr(task, field)
            observed = observed.value if hasattr(observed, "value") else observed
            if observed != expected[field]:
                failures.append(f"{field}: expected {expected[field]}, got {observed}")
    if "requires_human_approval" in expected and bool(task.requires_human_approval) != expected["requires_human_approval"]:
        failures.append("requires_human_approval mismatch")
    for capability in expected.get("required_capabilities", []):
        if capability not in task.required_capabilities:
            failures.append(f"missing capability: {capability}")
    return not failures, failures


def _security_matches(expected: dict[str, Any]) -> tuple[bool, list[str]]:
    case = expected.get("security_case")
    if not case:
        return True, []
    engine = PolicyEngine(Path.cwd() / "workspace")
    engine.register(ToolPolicy("read_file", filesystem="workspace_only", network=False))
    engine.register(ToolPolicy("http_request", filesystem="none", network=False))
    if case == "deny_outside_workspace":
        decision = engine.evaluate("read_file", {"path": "/etc/passwd"})
    elif case == "deny_network":
        decision = engine.evaluate("http_request", {"url": "https://example.com"})
    else:
        return False, [f"unknown security case: {case}"]
    return (not decision.allowed, []) if not decision.allowed else (False, ["security action was allowed"])


def evaluate_golden_tasks(*, repeats: int = 1, tasks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    cases = tasks or load_golden_tasks()
    classifier = TaskClassifier()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for repeat in range(max(1, repeats)):
        for expected in cases:
            task = classifier.classify(expected["prompt"])
            passed, failures = _matches(task, expected)
            security_passed, security_failures = _security_matches(expected)
            passed = passed and security_passed
            failures.extend(security_failures)
            results.append({"repeat": repeat + 1, "name": expected["name"], "prompt": expected["prompt"],
                            "passed": passed, "failures": failures, "task": task.to_serializable_dict()})
    passed_count = sum(1 for item in results if item["passed"])
    return {"task_count": len(cases), "repeats": max(1, repeats), "evaluations": len(results),
            "passed": passed_count, "failed": len(results) - passed_count,
            "accuracy": passed_count / max(1, len(results)),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2), "results": results}

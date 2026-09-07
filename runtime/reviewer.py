"""Independent deterministic review, with an optional local model hook."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


class ReviewResult(BaseModel):
    passed: bool
    changed_files: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    summary: str = ""
    reviewer_model: str = "deterministic"


def deterministic_review(*, workspace_root: str, result: dict[str, Any],
                         verification: dict[str, Any] | None = None) -> ReviewResult:
    changed = list(result.get("changed_files") or result.get("metadata", {}).get("changed_files", []) or [])
    issues: list[str] = []
    root = Path(workspace_root).resolve()
    for item in changed:
        path = (root / item).resolve()
        if root not in path.parents and path != root:
            issues.append(f"changed path escapes workspace: {item}")
        elif not path.exists():
            issues.append(f"changed file missing after mutation: {item}")
    check = verification or {}
    if check and check.get("passed") is False:
        issues.append("deterministic verification did not pass")
    passed = not issues and (not changed or bool(check.get("passed", True)))
    return ReviewResult(passed=passed, changed_files=changed, issues=issues,
                        summary="independent deterministic review passed" if passed else "; ".join(issues))

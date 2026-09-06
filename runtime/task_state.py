"""Typed contracts for bounded plan/execute/verify runs.

These contracts are deliberately framework-neutral so the existing LangGraph
state can carry them without serializing arbitrary model objects.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict, Annotated


def _append(existing: list[Any] | None, new: list[Any] | None) -> list[Any]:
    return [*(existing or []), *(new or [])]


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNED = "planned"
    RUNNING = "running"
    VERIFYING = "verifying"
    HEALING = "healing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    ABORTED = "aborted"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskStep(TypedDict, total=False):
    step_id: str
    ordinal: int
    description: str
    tool_name: str
    parameters: dict[str, Any]
    expected_evidence: list[dict[str, Any]]
    status: str
    result: dict[str, Any]
    error: dict[str, Any]
    retry_count: int
    max_retries: int
    repair_history: list[dict[str, Any]]
    started_at: str
    completed_at: str


class VerificationEvidence(TypedDict, total=False):
    kind: str
    passed: bool
    source: str
    detail: str
    path: str
    expected: Any
    observed: Any


class TaskRunState(TypedDict, total=False):
    run_id: str
    thread_id: str
    original_request: str
    normalized_request: str
    preprocessing: dict[str, Any]
    task: dict[str, Any]
    workspace_root: str
    selected_agent: str
    selected_model: str
    status: str
    plan: list[TaskStep]
    current_step_index: int
    task_retry_count: int
    max_task_retries: int
    max_step_retries: int
    max_tool_calls: int
    tool_call_count: int
    mutation_count: int
    messages: list[Any]
    step_results: Annotated[list[dict[str, Any]], _append]
    verification: dict[str, Any]
    evidence: list[VerificationEvidence]
    repair_history: Annotated[list[dict[str, Any]], _append]
    changed_files: list[str]
    baseline_snapshot: dict[str, Any]
    memory_facts: list[dict[str, Any]]
    # Bounded specialist working memory. This is execution state, not merely
    # an audit trail, and survives a graph-level repair invocation.
    agent_memory: dict[str, Any]
    trace: Annotated[list[dict[str, Any]], _append]
    errors: Annotated[list[dict[str, Any]], _append]
    final_answer: str

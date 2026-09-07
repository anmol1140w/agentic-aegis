"""Versioned contracts shared by planning, execution, review, and recovery."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionBudgets(BaseModel):
    """Hard limits for a single task run."""

    model_config = ConfigDict(extra="forbid")
    max_tool_calls: int = Field(default=40, ge=1, le=200)
    max_mutations: int = Field(default=10, ge=0, le=50)
    max_repairs: int = Field(default=3, ge=0, le=10)
    max_prompt_chars: int = Field(default=48_000, ge=2_000, le=200_000)
    max_file_chars: int = Field(default=12_000, ge=500, le=100_000)
    max_output_chars: int = Field(default=12_000, ge=500, le=100_000)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str
    objective: str
    required_evidence: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    expected_result: str = ""
    verification_condition: str = ""
    failure_strategy: str = "stop"
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=1, ge=0, le=3)


class FailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal[
        "SYNTAX_ERROR", "TEST_FAILURE", "IMPORT_ERROR", "TYPE_ERROR",
        "COMMAND_FAILURE", "PATH_ERROR", "PERMISSION_ERROR", "TIMEOUT",
        "DEPENDENCY_ERROR", "ENVIRONMENT_ERROR", "LOGIC_ERROR",
        "SECURITY_BLOCK", "UNKNOWN"
    ] = "UNKNOWN"
    code: str = "tool_failure"
    command: str = ""
    file: str = ""
    line: int | None = None
    assertion: str = ""
    stdout_summary: str = ""
    stderr_summary: str = ""
    likely_targets: list[str] = Field(default_factory=list)
    retryable: bool = True
    repair_strategy: str = "inspect_and_retry"


class EvidenceState(BaseModel):
    model_config = ConfigDict(extra="allow")
    changed_files: list[str] = Field(default_factory=list)
    tests_run: list[str] = Field(default_factory=list)
    tests_passed: list[str] = Field(default_factory=list)
    verification: list[dict[str, Any]] = Field(default_factory=list)
    remaining_risks: list[str] = Field(default_factory=list)


class TaskStateContract(BaseModel):
    """Serializable state contract; LangGraph carries its dictionary form."""

    model_config = ConfigDict(extra="allow")
    schema_version: int = 1
    task_id: str
    original_request: str
    normalized_request: str = ""
    repository_root: str = "."
    repository_map: dict[str, Any] = Field(default_factory=dict)
    relevant_files: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    plan: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    observations: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)
    patches: list[dict[str, Any]] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    test_results: list[dict[str, Any]] = Field(default_factory=list)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint_id: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    budgets: ExecutionBudgets = Field(default_factory=ExecutionBudgets)
    escalation_level: int = Field(default=0, ge=0, le=5)
    status: str = "pending"


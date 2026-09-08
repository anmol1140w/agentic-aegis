"""Typed workflow contracts and validation before execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from pydantic import BaseModel, Field, model_validator


class Capability(StrEnum):
    RAG = "rag"
    REASONING = "reasoning"
    CODING = "coding"
    VISION = "vision"
    DOCUMENT = "document_analysis"
    VERIFICATION = "verification"
    TOOL = "tool"


class PlanStep(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    capability: Capability
    input: str | None = None
    model: str | None = None
    tool: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoalBudget(BaseModel):
    """Hard resource limits for goal pursuit and recovery."""
    max_steps: int = Field(default=32, ge=1, le=10_000)
    max_tool_calls: int = Field(default=64, ge=0, le=100_000)
    max_retries: int = Field(default=2, ge=0, le=100)
    max_wall_time_seconds: float = Field(default=900.0, gt=0, le=86_400)
    max_tokens: int | None = Field(default=None, ge=1, le=10_000_000)


class GoalSpec(BaseModel):
    """Human-owned objective, acceptance criteria, constraints, and budget."""
    objective: str = Field(min_length=1, max_length=20_000)
    success_criteria: list[str] = Field(default_factory=list, max_length=100)
    constraints: list[str] = Field(default_factory=list, max_length=100)
    budget: GoalBudget = Field(default_factory=GoalBudget)
    risk: str = Field(default="low", pattern=r"^(low|medium|high|critical)$")
    human_authority: str = "human remains accountable for high-impact actions"


class ExecutionPlan(BaseModel):
    """A serialisable, dependency-aware execution plan."""

    version: str = "1"
    task_id: str = ""
    steps: list[PlanStep] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    goal: GoalSpec | None = None

    @model_validator(mode="after")
    def unique_ids(self) -> "ExecutionPlan":
        ids = [step.id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("execution plan contains duplicate step ids")
        if self.goal and len(self.steps) > self.goal.budget.max_steps:
            raise ValueError("execution plan exceeds the goal step budget")
        return self


@dataclass
class PlanValidator:
    """Validates graph shape, capabilities, models and tools before running."""

    model_capabilities: dict[str, set[str]] = field(default_factory=dict)
    allowed_models: set[str] | None = None
    allowed_tools: set[str] | None = None

    def validate(self, plan: ExecutionPlan) -> list[str]:
        errors: list[str] = []
        ids = {step.id for step in plan.steps}
        edges = {step.id: set(step.depends_on) for step in plan.steps}
        for step in plan.steps:
            missing = edges[step.id] - ids
            if missing:
                errors.append(f"{step.id}: unknown dependencies: {sorted(missing)}")
            if self.allowed_models is not None and step.model and step.model not in self.allowed_models:
                errors.append(f"{step.id}: model is not allowed: {step.model}")
            if self.allowed_tools is not None and step.tool and step.tool not in self.allowed_tools:
                errors.append(f"{step.id}: tool is not allowed: {step.tool}")
            if step.model and step.model in self.model_capabilities:
                caps = self.model_capabilities[step.model]
                if step.capability.value not in caps:
                    errors.append(f"{step.id}: {step.model} lacks {step.capability.value}")
        # Kahn's algorithm catches cycles and also documents the execution order.
        incoming = {node: len(deps & ids) for node, deps in edges.items()}
        ready = [node for node, count in incoming.items() if count == 0]
        visited = 0
        while ready:
            node = ready.pop()
            visited += 1
            for child, deps in edges.items():
                if node in deps:
                    incoming[child] -= 1
                    if incoming[child] == 0:
                        ready.append(child)
        if visited != len(plan.steps):
            errors.append("execution plan contains a dependency cycle")
        return errors

    def require_valid(self, plan: ExecutionPlan) -> None:
        errors = self.validate(plan)
        if errors:
            raise ValueError("invalid execution plan: " + "; ".join(errors))


@dataclass
class WorkflowRegistry:
    """Versioned workflow definitions, suitable for a visual editor/API."""

    _items: dict[str, dict[str, ExecutionPlan]] = field(default_factory=dict)

    def register(self, name: str, plan: ExecutionPlan, *, version: str | None = None) -> ExecutionPlan:
        version = version or plan.version
        self._items.setdefault(name, {})[version] = plan.model_copy(update={"version": version})
        return self._items[name][version]

    def get(self, name: str, version: str | None = None) -> ExecutionPlan:
        versions = self._items[name]
        return versions[version or sorted(versions)[-1]]

    def list(self) -> list[dict[str, str]]:
        return [{"name": name, "version": version} for name, versions in self._items.items() for version in versions]

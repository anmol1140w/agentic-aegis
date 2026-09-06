"""Strict, tolerant adapter for model-generated execution plans."""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from runtime.regex_safety import bounded_text


class PlanValidationError(ValueError):
    """A plan cannot be safely validated for execution."""


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    tool_name: str = Field(min_length=1, max_length=100)
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_evidence: list[dict[str, Any]] = Field(default_factory=list)
    max_retries: int = Field(default=1, ge=0, le=3)

    @field_validator("parameters")
    @classmethod
    def json_parameters(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must be JSON serializable") from exc
        return value


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal: str = Field(min_length=1, max_length=1000)
    steps: list[PlanStep] = Field(min_length=1, max_length=20)


def _extract_object(raw: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", bounded_text(raw), flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(text)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            for index, char in enumerate(candidate):
                if char == "{":
                    try:
                        value, _ = decoder.raw_decode(candidate[index:])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        continue
    raise PlanValidationError("model did not return a JSON plan object")


def parse_plan(raw: str, *, registered_tools: set[str], max_steps: int = 20) -> TaskPlan:
    """Parse and validate a plan without executing any content."""
    try:
        payload = _extract_object(raw)
        plan = TaskPlan.model_validate(payload)
    except Exception as exc:
        if isinstance(exc, PlanValidationError):
            raise
        raise PlanValidationError(str(exc)) from exc
    if len(plan.steps) > max_steps:
        raise PlanValidationError(f"plan exceeds maximum of {max_steps} steps")
    seen: set[str] = set()
    for step in plan.steps:
        if step.step_id in seen:
            raise PlanValidationError(f"duplicate step_id: {step.step_id}")
        seen.add(step.step_id)
        if step.tool_name not in registered_tools:
            raise PlanValidationError(f"unknown tool: {step.tool_name}")
    return plan

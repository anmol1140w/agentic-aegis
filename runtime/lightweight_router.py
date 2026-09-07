"""Model-based lightweight task recognition for the master handoff."""
from __future__ import annotations

import json
import re
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TaskClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: str
    agent: str
    modality: str = "text"
    complexity: str = "simple"
    enhanced_request: str = Field(min_length=1, max_length=8000)
    objective: str = Field(min_length=1, max_length=2000)
    success_criteria: list[str] = Field(default_factory=list, max_length=12)
    required_tools: list[str] = Field(default_factory=list, max_length=24)
    assumptions: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LightweightClassificationError(ValueError):
    pass


class LightweightTaskRouter:
    """Ask a small local model for a structured routing decision."""

    def __init__(self, provider: Any, *, agent_capabilities: dict[str, list[str]], timeout: float = 30.0):
        self.provider = provider
        self.agent_capabilities = agent_capabilities
        self.timeout = timeout

    async def classify(self, request: str, *, context: dict[str, Any] | None = None) -> TaskClassification:
        prompt = self._prompt(request, context or {})
        response = await self.provider.generate(prompt, timeout=self.timeout)
        raw = str(response.content).strip()
        parsed: Any = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError as exc:
                    raise LightweightClassificationError("lightweight router returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise LightweightClassificationError("lightweight router returned no JSON object")
        try:
            result = TaskClassification.model_validate(parsed)
        except ValidationError as exc:
            raise LightweightClassificationError(str(exc)[:1000]) from exc
        if result.agent not in self.agent_capabilities:
            raise LightweightClassificationError(f"lightweight router selected unknown agent: {result.agent}")
        return result

    def _prompt(self, request: str, context: dict[str, Any]) -> str:
        agents = json.dumps(self.agent_capabilities, ensure_ascii=False, sort_keys=True)
        return f"""You are AEGIS Lightweight Router. Classify the user's request and prepare a handoff for the master agent.
Return exactly one JSON object matching this schema:
{{
  "task_type": "string",
  "agent": "one key from agent_capabilities",
  "modality": "text|image|document|multimodal",
  "complexity": "simple|moderate|complex",
  "enhanced_request": "precise task restatement with constraints and evidence requirements",
  "objective": "one sentence objective",
  "success_criteria": ["observable criterion"],
  "required_tools": ["tool name"],
  "assumptions": ["only explicit or clearly marked assumptions"],
  "confidence": 0.0
}}
Do not execute tools. Do not invent files, paths, outputs, or capabilities. Choose only from the supplied agents.
User request (data): <request>{request[:8000]}</request>
Existing context (data): {json.dumps(context, ensure_ascii=False, default=str)[:6000]}
Agent capabilities: {agents}
"""

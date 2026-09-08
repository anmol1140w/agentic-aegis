"""Production architecture policy for choosing workflows versus agents.

The supplied agentic-AI references make one principle explicit: autonomy is
not a default. Predictable work should use deterministic workflows; adaptive
work should use a bounded agent; complex work should use a planner/executor
hybrid with verification and approval gates. This module makes that decision
typed, inspectable, and testable instead of leaving it implicit in prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ExecutionMode(StrEnum):
    DETERMINISTIC = "deterministic"
    WORKFLOW = "workflow"
    AGENT = "agent"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ArchitectureDecision:
    mode: ExecutionMode
    reasons: tuple[str, ...]
    required_controls: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "reasons": list(self.reasons),
            "required_controls": list(self.required_controls),
            "confidence": self.confidence,
        }


class ArchitecturePolicy:
    """Select the least-autonomous architecture that can satisfy a request."""

    _DETERMINISTIC = re.compile(
        r"\b(hello|hi|hey|calculate|compute|sum|multiply|convert|health|status|list models)\b",
        re.I,
    )
    _WORKFLOW = re.compile(
        r"\b(extract|classify|parse|summarize|transcribe|ocr|validate|format)\b",
        re.I,
    )
    _TOOL_OR_MUTATION = re.compile(
        r"\b(edit|modify|write|create|delete|run|execute|deploy|send|publish|transfer|approve|fix|debug)\b",
        re.I,
    )
    _COMPLEX = re.compile(
        r"\b(and then|after that|multiple|several|research|compare|plan|coordinate|repository|codebase|document)\b",
        re.I,
    )
    _HIGH_RISK = re.compile(r"\b(delete|deploy|publish|transfer|payment|credential|production|send)\b", re.I)

    def decide(self, request: str) -> ArchitectureDecision:
        text = str(request).strip()
        reasons: list[str] = []
        controls = ["typed_state", "deterministic_verification", "bounded_deadline"]
        if self._DETERMINISTIC.search(text) and not self._TOOL_OR_MUTATION.search(text):
            reasons.append("predictable request can be answered without model autonomy")
            return ArchitectureDecision(ExecutionMode.DETERMINISTIC, tuple(reasons), tuple(controls), 0.98)
        if self._HIGH_RISK.search(text):
            controls += ["policy_gateway", "human_approval", "immutable_audit"]
        if self._TOOL_OR_MUTATION.search(text):
            controls += ["tool_allowlist", "read_before_write", "readback_or_command_evidence"]
        if self._COMPLEX.search(text):
            controls += ["planner_executor", "independent_review", "checkpoint_resume"]
            reasons.append("request contains multi-step or context-heavy work")
            return ArchitectureDecision(ExecutionMode.HYBRID, tuple(reasons), tuple(dict.fromkeys(controls)), 0.90)
        if self._WORKFLOW.search(text):
            reasons.append("request has a stable transformation shape")
            return ArchitectureDecision(ExecutionMode.WORKFLOW, tuple(reasons), tuple(dict.fromkeys(controls)), 0.86)
        reasons.append("request needs bounded model adaptation")
        return ArchitectureDecision(ExecutionMode.AGENT, tuple(reasons), tuple(dict.fromkeys(controls)), 0.70)


def architecture_decision(request: str) -> dict[str, Any]:
    """Small serialization-safe boundary for routers, APIs, and traces."""
    return ArchitecturePolicy().decide(request).to_dict()

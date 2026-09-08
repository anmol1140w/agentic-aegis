"""Evaluation records for routing, tools, recovery, quality and latency."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvaluationRecord:
    task_id: str
    expected: Any
    actual: Any
    correctness: float
    latency_ms: float = 0.0
    token_usage: int = 0
    tool_errors: int = 0
    recovered: bool = False
    hallucination: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryRecord:
    """Evaluation dimensions for agent behavior, not only final answers."""
    task_id: str
    expected_tools: list[str] = field(default_factory=list)
    actual_tools: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: int = 0
    completed: bool = False
    policy_violations: int = 0
    retries: int = 0

    def score(self) -> dict[str, float]:
        expected = self.expected_tools
        actual = self.actual_tools
        correct_tools = sum(tool in expected for tool in actual) if expected else 1
        precision = correct_tools / len(actual) if actual else (1.0 if not expected else 0.0)
        recall = sum(tool in actual for tool in expected) / len(expected) if expected else 1.0
        return {
            "completion": float(self.completed),
            "tool_precision": precision,
            "tool_recall": recall,
            "tool_efficiency": 1.0 / max(1, len(actual)),
            "latency_ms": self.latency_ms,
            "tokens": float(self.tokens),
            "policy_violations": float(self.policy_violations),
            "retries": float(self.retries),
        }


class EvaluationSuite:
    def __init__(self, path: str | Path = ".aegis/evaluations.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[EvaluationRecord] = []

    def record(self, item: EvaluationRecord) -> EvaluationRecord:
        self.records.append(item)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item.__dict__, default=str) + "\n")
        return item

    def summary(self) -> dict[str, float]:
        if not self.records:
            return {"count": 0.0}
        return {
            "count": float(len(self.records)),
            "correctness": statistics.fmean(x.correctness for x in self.records),
            "latency_ms": statistics.fmean(x.latency_ms for x in self.records),
            "tool_error_rate": statistics.fmean(x.tool_errors > 0 for x in self.records),
            "recovery_rate": statistics.fmean(x.recovered for x in self.records),
            "hallucination_rate": statistics.fmean(x.hallucination for x in self.records),
        }

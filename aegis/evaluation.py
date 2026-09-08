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

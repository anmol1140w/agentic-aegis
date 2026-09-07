"""Deterministic, offline benchmark scoring; execution is injectable."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Any


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str
    difficulty: str
    expected_files: tuple[str, ...] = ()


@dataclass
class BenchmarkResult:
    case_id: str
    mode: str
    success: bool = False
    first_attempt_success: bool = False
    repair_success: bool = False
    false_completion: bool = False
    tool_calls: int = 0
    prompt_chars: int = 0
    latency_ms: float = 0.0
    mutation_count: int = 0
    rollback_count: int = 0
    sandbox_violations: int = 0
    approval_interruptions: int = 0


def run_cases(cases: list[BenchmarkCase], runner: Callable[[BenchmarkCase, str], dict[str, Any]]) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for mode in ("coder_only", "aegis"):
        for case in cases:
            raw = runner(case, mode)
            results.append(BenchmarkResult(case_id=case.case_id, mode=mode, **{
                key: raw[key] for key in BenchmarkResult.__dataclass_fields__
                if key != "case_id" and key != "mode" and key in raw
            }))
    return results


def summarize(results: list[BenchmarkResult]) -> dict[str, Any]:
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)
    report: dict[str, Any] = {}
    for mode, items in grouped.items():
        count = max(1, len(items))
        report[mode] = {
            "cases": len(items),
            "success_rate": sum(x.success for x in items) / count,
            "first_attempt_rate": sum(x.first_attempt_success for x in items) / count,
            "repair_success_rate": sum(x.repair_success for x in items) / count,
            "false_completion_rate": sum(x.false_completion for x in items) / count,
            "avg_tool_calls": sum(x.tool_calls for x in items) / count,
            "avg_prompt_chars": sum(x.prompt_chars for x in items) / count,
            "avg_latency_ms": sum(x.latency_ms for x in items) / count,
            "mutations": sum(x.mutation_count for x in items),
            "rollbacks": sum(x.rollback_count for x in items),
            "sandbox_violations": sum(x.sandbox_violations for x in items),
            "approval_interruptions": sum(x.approval_interruptions for x in items),
        }
    return report

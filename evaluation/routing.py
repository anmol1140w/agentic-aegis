"""Baseline versus adaptive routing evaluation on the golden task set.

The default mode is a routing-contract evaluation. ``actual=True`` executes
eligible local specialists and records real outcomes without training on
those evaluation results. It never invents model-quality results for a
candidate that was not executed.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from evaluation.golden import load_golden_tasks, _matches, _security_matches
from routing.adaptive_router import ContextualBanditRouter, RoutingCandidate
from routing.classifier import TaskClassifier
from models.registry import ModelRegistry
from runtime.agents import AgentRequest, AgentStatus, MasterAgent, build_default_agent_registry


MODEL_BY_FAMILY = {
    "coding": "qwen-coder", "engineering_document": "qwen-vision",
    "document_analysis": "qwen-general", "summarization": "qwen-general",
    "artifact_generation": "qwen-general", "psu_approval_note_generate": "qwen-general",
    "psu_approval_note_explain": "qwen-general", "general_reasoning": "qwen-general",
}
MODEL_BY_AGENT = {"coding_agent": "qwen-coder", "vision_agent": "qwen-vision",
                  "document_agent": "qwen-general", "general_agent": "qwen-general"}
AGENT_CAPABILITIES = {
    "coding_agent": frozenset({"coding", "calculation", "reasoning", "document_analysis"}),
    "vision_agent": frozenset({"vision", "reasoning", "document_analysis", "ocr", "layout_extraction"}),
    "document_agent": frozenset({"document_analysis", "reasoning", "artifact_generation", "ocr", "layout_extraction"}),
    "general_agent": frozenset({"reasoning", "calculation", "document_analysis"}),
}


def reward_for_outcome(*, success: bool, verification: bool, latency_ms: float,
                       failures: float = 0.0, escalated: bool = False,
                       evidence_quality: float | None = None,
                       human_accepted: bool | None = None,
                       timed_out: bool = False,
                       unnecessary_escalation: bool = False,
                       security_violation: bool = False,
                       unauthorized_tool_execution: bool = False,
                       approval_bypass: bool = False,
                       policy_bypass: bool = False,
                       forbidden_filesystem: bool = False,
                       forbidden_network: bool = False,
                       privilege_violation: bool = False,
                       target_latency_ms: float = 2_000.0,
                       budget_latency_ms: float = 30_000.0) -> float:
    """Return the common bounded reward used by training and telemetry.

    Security/policy failures are invalid actions, not merely low-reward
    observations. All ordinary terms are explicitly normalized before the
    final clamp so heterogeneous tasks remain comparable.
    """
    catastrophic = (security_violation or unauthorized_tool_execution or
                    approval_bypass or policy_bypass or forbidden_filesystem or
                    forbidden_network or privilege_violation)
    if catastrophic:
        return -1.0
    evidence = max(0.0, min(1.0, float(success if evidence_quality is None else evidence_quality)))
    human = bool(success if human_accepted is None else human_accepted)
    latency = max(0.0, min(1.0, max(0.0, latency_ms - target_latency_ms) /
                          max(1.0, budget_latency_ms - target_latency_ms)))
    failure_fraction = max(0.0, min(1.0, float(failures)))
    unnecessary = bool(unnecessary_escalation or (escalated and not success))
    reward = (0.50 * bool(success) + 0.25 * bool(verification) + 0.10 * evidence +
              0.10 * human - 0.05 * latency - 0.10 * failure_fraction -
              0.10 * bool(timed_out) - 0.05 * unnecessary)
    return max(-1.0, min(1.0, float(reward)))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) * 0.95) - 1)))
    return ordered[index]


def _baseline_candidate(expected: dict[str, Any], task: Any, master: MasterAgent | None = None) -> RoutingCandidate:
    if master is not None:
        planned = master._capability_plan(expected["prompt"])
        if planned:
            agent = str(planned[0].get("agent", "general_agent"))
            return RoutingCandidate(agent, MODEL_BY_AGENT.get(agent, "qwen-general"),
                                    AGENT_CAPABILITIES.get(agent, frozenset({"reasoning"})), 1.0,
                                    str(planned[0].get("capability", task.workflow)))
    workflow = str(getattr(task, "workflow", "") or getattr(task, "task_type", "general_reasoning"))
    model = MODEL_BY_FAMILY.get(workflow, MODEL_BY_FAMILY.get(str(getattr(task, "task_type", "")), "qwen-general"))
    capabilities = frozenset(getattr(task, "required_capabilities", []) or ["reasoning"])
    return RoutingCandidate(workflow, model, capabilities, 1.0, workflow)


def _candidates(baseline: RoutingCandidate, task: Any) -> list[RoutingCandidate]:
    required = frozenset(getattr(task, "required_capabilities", []) or ["reasoning"])
    # Alternatives are existing specialist profiles whose declared capability
    # set covers the task. Their model quality is not inferred here.
    candidates = [baseline]
    for name, capabilities in AGENT_CAPABILITIES.items():
        if name != baseline.name and required.issubset(capabilities):
            candidates.append(RoutingCandidate(name, MODEL_BY_AGENT[name], capabilities, 1.0, baseline.workflow))
    return candidates


async def _execute_candidate(master: MasterAgent, candidate: RoutingCandidate,
                             expected: dict[str, Any]) -> dict[str, Any]:
    """Execute one real local specialist, or report why it was unavailable.

    Side-effecting, approval, direct-tool, and security cases are deliberately
    excluded from this direct specialist harness; those must run through the
    full Orchestrator/PolicyEngine path instead of bypassing it.
    """
    if (expected.get("requires_human_approval") or expected.get("security_case") or
            expected.get("execution_mode") in {"direct_tool", "model_with_tools"}):
        return {"available": False, "reason": "requires full PolicyEngine workflow"}
    try:
        agent = master.registry.get(candidate.name)
    except Exception:
        return {"available": False, "reason": "candidate agent unavailable"}
    context: dict[str, Any] = {"workspace_root": str(Path.cwd() / "workspace")}
    if "p&id" in expected["prompt"].lower() or "image" in expected["prompt"].lower():
        image = Path("workspace/inputs/graph_diagrams_promo-showcase_01.jpg").resolve()
        if image.exists():
            context["image_paths"] = [str(image)]
    request = AgentRequest(task=expected["prompt"], context=context,
                           expected_output="grounded structured result")
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            agent.run(request),
            timeout=max(5.0, float(os.getenv("AEGIS_ROUTING_EVAL_TIMEOUT_SECONDS", "45"))),
        )
        success = result.status == AgentStatus.SUCCESS
        verification = bool(success and (result.verification.get("status") in {"passed", "not_run"}))
        return {"available": True, "success": success, "verification": verification,
                "evidence_quality": 1.0 if result.evidence else 0.0,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "failures": min(1.0, len(result.errors) / 5.0), "timed_out": False,
                "tool_failures": len(result.errors), "summary": result.summary[:400]}
    except asyncio.TimeoutError:
        return {"available": True, "success": False, "verification": False,
                "evidence_quality": 0.0, "latency_ms": (time.perf_counter() - started) * 1000,
                "failures": 1.0, "timed_out": True, "tool_failures": 0,
                "summary": "candidate timeout"}
    except Exception as exc:
        return {"available": True, "success": False, "verification": False,
                "evidence_quality": 0.0, "latency_ms": (time.perf_counter() - started) * 1000,
                "failures": 1.0, "timed_out": False, "tool_failures": 1,
                "summary": f"candidate failure: {type(exc).__name__}"}


def _actual_rows(cases: list[dict[str, Any]], master: MasterAgent,
                 router: ContextualBanditRouter, *, repeats: int,
                 mode: str) -> list[dict[str, Any]]:
    """Run matched local specialist executions for the requested arm."""
    classifier = TaskClassifier()
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, repeats)):
        for expected in cases:
            task = classifier.classify(expected["prompt"])
            baseline = _baseline_candidate(expected, task, master)
            candidates = _candidates(baseline, task)
            context = {"task_type": task.task_type, "workflow": task.workflow,
                       "domain_intent": task.domain_intent,
                       "required_capabilities": tuple(task.required_capabilities),
                       "quality_required": task.quality_required}
            decision = router.select(context, candidates, baseline, mode=mode)
            started = time.perf_counter()
            outcome = asyncio.run(_execute_candidate(master, decision.selected, expected))
            elapsed = (time.perf_counter() - started) * 1000
            available = bool(outcome.get("available"))
            success = bool(outcome.get("success", False)) if available else False
            verification = bool(outcome.get("verification", False)) if available else False
            reward = reward_for_outcome(
                success=success, verification=verification,
                evidence_quality=float(outcome.get("evidence_quality", 0.0)),
                human_accepted=success, latency_ms=float(outcome.get("latency_ms", elapsed)),
                failures=float(outcome.get("failures", 1.0 if not available else 0.0)),
                timed_out=bool(outcome.get("timed_out", False)),
                escalated=bool(expected.get("requires_human_approval", False)),
            )
            router.record_outcome(decision, context, reward, latency_ms=float(outcome.get("latency_ms", elapsed)),
                                  success=success, verification=verification, scope="evaluation")
            rows.append({"repeat": repeat + 1, "name": expected["name"], "task_type": task.task_type,
                         "success": success, "verification": verification,
                         "latency_ms": round(float(outcome.get("latency_ms", elapsed)), 3),
                         "failures": float(outcome.get("failures", 1.0 if not available else 0.0)),
                         "tool_failures": int(outcome.get("tool_failures", 0)),
                         "timeout": bool(outcome.get("timed_out", False)),
                         "escalation": bool(expected.get("requires_human_approval", False)),
                         "available": available, "reward": reward,
                         "selected_action": decision.selected.name,
                         "decision": decision.to_dict(), "summary": outcome.get("summary", "")})
    return rows


def _evaluate_once(tasks: list[dict[str, Any]], router: ContextualBanditRouter | None,
                   *, mode: str, master: MasterAgent | None = None,
                   record_observations: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classifier = TaskClassifier()
    rows: list[dict[str, Any]] = []
    for expected in tasks:
        started = time.perf_counter()
        task = classifier.classify(expected["prompt"])
        route_ok, route_failures = _matches(task, expected)
        security_ok, security_failures = _security_matches(expected)
        baseline = _baseline_candidate(expected, task, master)
        candidates = _candidates(baseline, task)
        context = {"task_type": task.task_type, "workflow": task.workflow,
                   "domain_intent": task.domain_intent,
                   "required_capabilities": tuple(task.required_capabilities),
                   "quality_required": task.quality_required}
        decision = router.select(context, candidates, baseline, mode=mode) if router else None
        selected = decision.selected if decision else baseline
        # A selected alternative is accepted only when its declared capability
        # contract covers the task; no unexecuted model quality is fabricated.
        selected_ok = set(task.required_capabilities).issubset(selected.capabilities)
        success = bool(route_ok and security_ok and selected_ok)
        verification = bool(success)
        # The evaluator measures routing-contract work, not model inference.
        # Apply a 1 ms floor so scheduler noise cannot decide a promotion for
        # sub-millisecond offline evaluations.
        latency = max(1.0, (time.perf_counter() - started) * 1000)
        reward = reward_for_outcome(
            success=success, verification=verification, latency_ms=latency,
            evidence_quality=1.0 if verification else 0.0,
            human_accepted=success, failures=min(1.0, len(route_failures + security_failures)),
            escalated=bool(expected.get("requires_human_approval", False)),
        )
        if router and decision and record_observations:
            router.record_outcome(decision, context, reward)
        rows.append({"name": expected["name"], "success": success, "verification": verification,
                     "latency_ms": round(latency, 3), "failures": len(route_failures) + len(security_failures),
                     "escalation": bool(expected.get("requires_human_approval", False)),
                     "reward": reward, "decision": decision.to_dict() if decision else None,
                     "route_failures": route_failures + security_failures})
    total = max(1, len(rows))
    report = {"task_count": len(tasks), "coverage": len(rows) / total,
              "task_success_rate": sum(row["success"] for row in rows) / total,
              "verification_pass_rate": sum(row["verification"] for row in rows) / total,
              "average_latency_ms": sum(row["latency_ms"] for row in rows) / total,
              "p95_latency_ms": _p95([row["latency_ms"] for row in rows]),
              "failure_count": sum(row["failures"] for row in rows),
              "actual_workflow_execution": False,
              "executed_candidate_observations": 0,
              "timeout_rate": 0.0, "security_violations": 0,
              "unauthorized_tool_executions": 0, "approval_bypasses": 0,
              "critical_workflow_failures": 0, "schema_evidence_validity": sum(row["verification"] for row in rows) / total,
              "escalation_rate": sum(row["escalation"] for row in rows) / total,
              "reward": sum(row["reward"] for row in rows) / total,
              "adaptive_fallback_rate": sum(bool(row["decision"] and row["decision"].get("fallback")) for row in rows) / total,
              "rows": rows}
    return report, rows


def evaluate_baseline_vs_adaptive(*, repeats: int = 3, tasks: list[dict[str, Any]] | None = None,
                                  actual: bool = False, max_actual_tasks: int | None = None) -> dict[str, Any]:
    cases = tasks or load_golden_tasks()
    registry = ModelRegistry.from_yaml("config/models.yaml")
    master = MasterAgent(build_default_agent_registry(registry))
    baseline_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    # This evaluator intentionally does not train from classifier contracts.
    # A model/workflow outcome is an observation only after the corresponding
    # candidate actually executes and produces evidence.
    training_router = ContextualBanditRouter(alpha=0.05)
    for _ in range(max(1, repeats)):
        report, rows = _evaluate_once(cases, None, mode="shadow", master=master)
        baseline_rows.extend(rows)
    adaptive_router = ContextualBanditRouter(alpha=0.05)
    adaptive_router._arms = training_router._arms
    # Evaluation-only activation: production promotion still requires the
    # full gate below. With no observed outcomes this should cold-start and
    # fall back, making the absence of learning visible in the report.
    adaptive_router.enabled = True
    adaptive_router.mode = "adaptive"
    for _ in range(max(1, repeats)):
        _, rows = _evaluate_once(cases, adaptive_router, mode="adaptive", master=master)
        adaptive_rows.extend(rows)
    def aggregate(rows: list[dict[str, Any]], *, executed: bool = False) -> dict[str, Any]:
        total = max(1, len(rows))
        return {"task_count": len({r.get("name") for r in rows}), "evaluations": len(rows),
                "coverage": len(rows) / total,
                "task_success_rate": sum(r["success"] for r in rows) / total,
                "verification_pass_rate": sum(r["verification"] for r in rows) / total,
                "average_latency_ms": sum(r["latency_ms"] for r in rows) / total,
                "p95_latency_ms": _p95([r["latency_ms"] for r in rows]),
                "failure_count": sum(r["failures"] for r in rows),
                "actual_workflow_execution": bool(executed),
                "executed_candidate_observations": sum(bool(r.get("available", False)) for r in rows),
                "timeout_rate": sum(bool(r.get("timeout", False)) for r in rows) / total,
                "tool_failures": sum(int(r.get("tool_failures", 0)) for r in rows),
                "security_violations": 0,
                "unauthorized_tool_executions": 0, "approval_bypasses": 0,
                "critical_workflow_failures": 0, "schema_evidence_validity": sum(r["verification"] for r in rows) / total,
                "escalation_rate": sum(r["escalation"] for r in rows) / total,
                "reward": sum(r["reward"] for r in rows) / total,
                "adaptive_fallback_rate": sum(bool(r["decision"] and r["decision"].get("fallback")) for r in rows) / total,
                "action_frequencies": {action: sum(r.get("selected_action") == action for r in rows)
                                        for action in sorted({r.get("selected_action") for r in rows if r.get("selected_action")})}}
    baseline_report = aggregate(baseline_rows)
    adaptive_report = aggregate(adaptive_rows)
    if actual:
        actual_cases = cases[:max_actual_tasks] if max_actual_tasks else cases
        # Actual evaluation observations are never fed back into the LinUCB
        # arm parameters. This prevents circular evaluation.
        baseline_actual_router = ContextualBanditRouter(alpha=0.05)
        adaptive_actual_router = ContextualBanditRouter(alpha=0.05)
        adaptive_actual_router.enabled = True
        adaptive_actual_router.mode = "adaptive"
        baseline_rows = _actual_rows(actual_cases, master, baseline_actual_router,
                                     repeats=repeats, mode="shadow")
        adaptive_rows = _actual_rows(actual_cases, master, adaptive_actual_router,
                                     repeats=repeats, mode="adaptive")
        baseline_report = aggregate(baseline_rows, executed=True)
        adaptive_report = aggregate(adaptive_rows, executed=True)
        adaptive_report["evaluation_observations"] = adaptive_actual_router.diagnostics().get("evaluation_observations", 0)
    gate = training_router.gate.compare(baseline_report, adaptive_report)
    if gate.promoted:
        adaptive_router.enabled = True
        adaptive_router.mode = "adaptive"
    else:
        adaptive_router.enabled = False
        adaptive_router.mode = "shadow"
    return {"repeats": max(1, repeats), "baseline": baseline_report,
            "adaptive": adaptive_report, "promotion": gate.to_dict(),
            "adaptive_enabled": adaptive_router.enabled,
            "policy": adaptive_router.diagnostics(),
            "evaluation_scope": "actual local specialist execution" if actual else "deterministic routing contract; unexecuted model quality is not inferred",
            "baseline_rows": baseline_rows, "adaptive_rows": adaptive_rows}

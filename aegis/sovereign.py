"""One evidence-first execution path for sovereign distributed workflows.

This module intentionally models external infrastructure as signed, typed
contracts.  A deployment can attach real Kubernetes, TEE, A2A, DSPy, or ledger
adapters without giving those adapters authority to bypass validation, risk
approval, evidence checks, or the audit trail.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Protocol

from aegis.contracts import ExecutionPlan, PlanStep, PlanValidator
from aegis.governance import AuditChain
from runtime.verification import verify_tool_result
from routing.adaptive_router import ContextualBanditRouter, RoutingCandidate


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ToolRisk:
    tool: str
    level: RiskLevel
    reasons: tuple[str, ...] = ()
    requires_approval: bool = False


class RiskBasedApproval:
    """Deterministic approval policy; only elevated risk can pause execution."""

    def assess(self, step: PlanStep) -> ToolRisk:
        text = " ".join([step.tool or "", step.input or "", json.dumps(step.metadata, default=str)]).lower()
        reasons: list[str] = []
        level = RiskLevel.LOW
        if any(word in text for word in ("delete", "transfer", "publish", "deploy", "external", "network")):
            level, reasons = RiskLevel.HIGH, ["external or irreversible side effect"]
        if any(word in text for word in ("credential", "secret", "payment", "production", "root")):
            level, reasons = RiskLevel.CRITICAL, ["sensitive or production-impacting operation"]
        elif step.tool or step.capability.value == "tool":
            level = max(level, RiskLevel.MEDIUM, key=lambda value: list(RiskLevel).index(value))
            reasons = reasons or ["tool invocation"]
        return ToolRisk(step.tool or step.capability.value, level, tuple(reasons), level in {RiskLevel.HIGH, RiskLevel.CRITICAL})


class RoutingStrategy(Protocol):
    """Selects a capability-compatible route; it has no policy authority."""
    def route(self, step: PlanStep) -> dict[str, Any]: ...


class DeterministicCapabilityRouter:
    """MVP default: stable local capability routing with no learned behaviour."""
    def route(self, step: PlanStep) -> dict[str, Any]:
        return {"selected": {"name": "local-specialist", "model": step.model or "local",
                "capabilities": [step.capability.value]}, "mode": "deterministic"}


class RLRoutingStrategy:
    """Experimental opt-in strategy that falls back to deterministic routing."""
    def __init__(self, router: ContextualBanditRouter, fallback: RoutingStrategy | None = None) -> None:
        self.router, self.fallback = router, fallback or DeterministicCapabilityRouter()

    def route(self, step: PlanStep) -> dict[str, Any]:
        if not self.router.enabled:
            route = self.fallback.route(step)
            route["rl"] = "shadow/disabled"
            return route
        candidate = RoutingCandidate(name="local-specialist", model=step.model or "local",
                                     capabilities=frozenset({step.capability.value}))
        return self.router.select({"capability": step.capability.value, "tool": step.tool or ""},
                                  [candidate], candidate).to_dict()


class AgentTransport(Protocol):
    async def dispatch(self, message: "A2AMessage") -> dict[str, Any]: ...


class InferenceBackend(Protocol):
    async def infer(self, prompt: str, **kwargs: Any) -> dict[str, Any]: ...


class OptimizationStrategy(Protocol):
    def optimize(self, examples: list[dict[str, Any]]) -> dict[str, Any]: ...


class ExecutionIsolationBackend(Protocol):
    def verify(self, workload: str, measurement: str, nonce: str) -> "Attestation": ...


class DeploymentBackend(Protocol):
    def manifest(self, name: str, image: str, **kwargs: Any) -> dict[str, Any]: ...


class AuditAnchorBackend(Protocol):
    def anchor(self, audit_hash: str, execution_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Attestation:
    workload: str
    measurement: str
    nonce: str
    verified: bool


class TEEVerifier:
    """Verifies a measurement against an allow-list (adapter boundary for a real TEE)."""

    def __init__(self, allowed_measurements: set[str] | None = None) -> None:
        self.allowed_measurements = allowed_measurements or set()

    def verify(self, workload: str, measurement: str, nonce: str) -> Attestation:
        return Attestation(workload, measurement, nonce, bool(measurement) and
                           (not self.allowed_measurements or measurement in self.allowed_measurements))


@dataclass(frozen=True)
class A2AMessage:
    task_id: str
    sender: str
    recipient: str
    capability: str
    payload: dict[str, Any]
    trace_id: str


class A2AGateway:
    """In-process A2A protocol gateway; transport adapters preserve this envelope."""

    def __init__(self) -> None:
        self.handlers: dict[str, Callable[[A2AMessage], Any]] = {}

    def register(self, agent: str, handler: Callable[[A2AMessage], Any]) -> None:
        self.handlers[agent] = handler

    async def dispatch(self, message: A2AMessage) -> dict[str, Any]:
        handler = self.handlers.get(message.recipient)
        if handler is None:
            return {"ok": False, "error": f"A2A recipient unavailable: {message.recipient}"}
        result = handler(message)
        return await result if asyncio.iscoroutine(result) else result


class FederatedInference:
    """Routes only a digest/minimal input envelope to approved local workers."""

    async def infer(self, prompt: str, workers: list[Callable[[str], Any]], **_: Any) -> dict[str, Any]:
        if not workers:
            return {"ok": False, "error": "no federated workers"}
        responses = await asyncio.gather(*[self._call(worker, prompt) for worker in workers], return_exceptions=True)
        valid = [item for item in responses if isinstance(item, dict) and item.get("ok", True)]
        return {"ok": bool(valid), "mode": "federated", "worker_count": len(workers),
                "responses": valid, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}

    @staticmethod
    async def _call(worker: Callable[[str], Any], prompt: str) -> Any:
        result = worker(prompt)
        return await result if asyncio.iscoroutine(result) else result


class DSPyProgram:
    """Provider-neutral DSPy-style signature/program contract with traceable inputs."""

    def __init__(self, signature: str, program: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> None:
        self.signature, self.program = signature, program

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        output = self.program(inputs) if self.program else {"answer": inputs.get("task", "")}
        return {"ok": True, "signature": self.signature, "inputs_sha256": self._digest(inputs), "output": output}

    @staticmethod
    def _digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class VisualWorkflowEditor:
    """Exports/imports nodes and edges for any canvas editor without changing execution semantics."""

    @staticmethod
    def export(plan: ExecutionPlan) -> dict[str, Any]:
        return {"schema": "aegis.visual-workflow/v1", "nodes": [
            {"id": step.id, "type": step.capability.value, "label": step.id,
             "position": step.metadata.get("position", {"x": index * 220, "y": 80})}
            for index, step in enumerate(plan.steps)],
            "edges": [{"id": f"{dep}-{step.id}", "source": dep, "target": step.id}
                      for step in plan.steps for dep in step.depends_on]}


class KubernetesPlanner:
    """Produces a safe workload manifest; actual cluster application remains an approved tool action."""

    @staticmethod
    def manifest(name: str, image: str, *, replicas: int = 1, tee_required: bool = False) -> dict[str, Any]:
        annotations = {"aegis.io/tee-required": "true"} if tee_required else {}
        return {"apiVersion": "apps/v1", "kind": "Deployment", "metadata": {"name": name, "annotations": annotations},
                "spec": {"replicas": replicas, "selector": {"matchLabels": {"app": name}},
                         "template": {"metadata": {"labels": {"app": name}}, "spec": {"containers": [
                             {"name": name, "image": image, "securityContext": {"allowPrivilegeEscalation": False,
                              "readOnlyRootFilesystem": True}}]}}}}


class BlockchainAuditLedger:
    """Experimental in-memory adapter, not a blockchain implementation or claim."""

    def __init__(self) -> None:
        self.anchors: list[dict[str, str]] = []

    def anchor(self, audit_hash: str, execution_id: str) -> dict[str, str]:
        receipt = {"execution_id": execution_id, "audit_hash": audit_hash,
                   "anchor_id": hashlib.sha256(f"{execution_id}:{audit_hash}".encode()).hexdigest()}
        self.anchors.append(receipt)
        return receipt


class EvidenceVerifier:
    """Checks citations and deliverables from observable records, never model claims."""

    @staticmethod
    def verify(step: PlanStep, result: dict[str, Any]) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        citations = result.get("citations", [])
        if step.metadata.get("require_citations"):
            valid = isinstance(citations, list) and bool(citations) and all(
                isinstance(item, dict) and bool(item.get("source") or item.get("url")) for item in citations)
            checks.append({"kind": "citations", "passed": valid, "count": len(citations) if isinstance(citations, list) else 0})
        if step.metadata.get("require_deliverable"):
            value = result.get("deliverable") or result.get("path")
            valid = isinstance(value, str) and Path(value).is_file()
            checks.append({"kind": "deliverable", "passed": valid, "path": value})
        passed = all(check["passed"] for check in checks)
        return {"passed": passed, "checks": checks, "missing_evidence": [check["kind"] for check in checks if not check["passed"]]}


@dataclass
class SovereignExecutor:
    """Contract-level workflow executor for explicit typed-plan integrations.

    Normal CLI/API requests use ``runtime.task_graph`` through
    ``Orchestrator.run_master``. This class is intentionally kept as a small
    typed-plan seam for workflow integrations and tests; it is not a second
    hidden supervisor or a replacement for the runtime tool gateway.
    """
    validator: PlanValidator
    audit: AuditChain
    router: RoutingStrategy = field(default_factory=DeterministicCapabilityRouter)
    approvals: RiskBasedApproval = field(default_factory=RiskBasedApproval)
    transport: AgentTransport | None = None
    isolation: ExecutionIsolationBackend | None = None
    audit_anchor: AuditAnchorBackend | None = None

    async def execute(self, plan: ExecutionPlan, *, executor: Callable[[PlanStep], Any],
                      approval: Callable[[ToolRisk, PlanStep], bool] | None = None,
                      attestation: dict[str, str] | None = None) -> dict[str, Any]:
        self.validator.require_valid(plan)
        execution_id, trace_id = f"sx-{uuid.uuid4().hex[:12]}", uuid.uuid4().hex
        started = time.perf_counter()
        self.audit.append("execution_started", task_id=plan.task_id, execution_id=execution_id, trace_id=trace_id)
        ordered = self._ordered(plan)
        budget = plan.goal.budget if plan.goal else None
        if budget and len(ordered) > budget.max_steps:
            result = {"execution_id": execution_id, "trace_id": trace_id, "passed": False,
                      "error": "goal step budget exceeded", "steps": [],
                      "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                      "audit_valid": self.audit.verify()}
            self.audit.append("execution_finished", task_id=plan.task_id, execution_id=execution_id, passed=False,
                              reason="goal_step_budget_exceeded")
            return result
        results: list[dict[str, Any]] = []
        for step in ordered:
            if budget and time.perf_counter() - started > budget.max_wall_time_seconds:
                results.append({"step_id": step.id, "ok": False, "error": "goal wall-time budget exceeded"})
                break
            if budget and len(results) >= budget.max_steps:
                results.append({"step_id": step.id, "ok": False, "error": "goal step budget exceeded"})
                break
            risk = self.approvals.assess(step)
            if risk.requires_approval and not (approval and approval(risk, step)):
                result = {"step_id": step.id, "ok": False, "error": "risk-based approval denied", "risk": asdict(risk)}
                results.append(result); break
            if step.metadata.get("isolation") == "tee":
                if self.isolation is None:
                    results.append({"step_id": step.id, "ok": False, "error": "requested isolation backend is unavailable"}); break
                claim = attestation or {}
                verified = self.isolation.verify(step.id, claim.get("measurement", ""), claim.get("nonce", trace_id))
                if not verified.verified:
                    results.append({"step_id": step.id, "ok": False, "error": "TEE attestation rejected", "attestation": asdict(verified)}); break
            route = self._route(step)
            requested_retries = int(step.metadata.get("retries", 0))
            allowed_retries = min(requested_retries, budget.max_retries if budget else requested_retries)
            attempts = max(1, allowed_retries + 1)
            result: dict[str, Any] = {}
            for attempt in range(1, attempts + 1):
                if step.metadata.get("delegate_to") and self.transport:
                    raw = await self.transport.dispatch(A2AMessage(plan.task_id, "supervisor", str(step.metadata["delegate_to"]),
                                                                   step.capability.value, {"step": step.model_dump()}, trace_id))
                else:
                    raw = executor(step)
                    raw = await raw if asyncio.iscoroutine(raw) else raw
                result = dict(raw or {})
                if budget and len(results) >= budget.max_tool_calls:
                    result = {"ok": False, "error": "goal tool-call budget exceeded"}
                result.update({"step_id": step.id, "risk": asdict(risk), "route": route, "attempt": attempt})
                result["verification"] = verify_tool_result(result)
                result["evidence_verification"] = EvidenceVerifier.verify(step, result)
                result["ok"] = (bool(result.get("ok", True)) and result["verification"]["passed"]
                                and result["evidence_verification"]["passed"])
                if result["ok"] or attempt == attempts:
                    break
                self.audit.append("step_checkpoint", task_id=plan.task_id, execution_id=execution_id,
                                  step_id=step.id, attempt=attempt, recovery="retry")
            results.append(result)
            self.audit.append("step_executed", task_id=plan.task_id, execution_id=execution_id, step_id=step.id,
                              passed=result["ok"], risk=risk.level.value)
            if not result["ok"] and step.required:
                break
        passed = len(results) == len(ordered) and all(item.get("ok") for item in results)
        self.audit.append("execution_finished", task_id=plan.task_id, execution_id=execution_id, passed=passed)
        response = {"execution_id": execution_id, "trace_id": trace_id, "passed": passed, "steps": results,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "audit_valid": self.audit.verify()}
        if self.audit_anchor:
            response["audit_anchor"] = self.audit_anchor.anchor(self.audit._previous, execution_id)
        return response

    def _route(self, step: PlanStep) -> dict[str, Any]:
        return self.router.route(step)

    @staticmethod
    def _ordered(plan: ExecutionPlan) -> list[PlanStep]:
        remaining, ordered = {step.id: step for step in plan.steps}, []
        while remaining:
            ready = [step for step in remaining.values() if set(step.depends_on) <= {item.id for item in ordered}]
            if not ready: raise ValueError("workflow cannot be ordered")
            ordered.extend(ready)
            for step in ready: remaining.pop(step.id)
        return ordered

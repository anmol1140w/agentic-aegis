from pathlib import Path

import pytest

from aegis.contracts import Capability, ExecutionPlan, GoalBudget, GoalSpec, PlanStep, PlanValidator
from aegis.governance import AuditChain
from aegis.sovereign import (A2AGateway, A2AMessage, DSPyProgram, FederatedInference,
                             KubernetesPlanner, SovereignExecutor, TEEVerifier,
                             VisualWorkflowEditor)


@pytest.mark.asyncio
async def test_sovereign_execution_normal_validates_routes_verifies_and_audits(tmp_path: Path):
    plan = ExecutionPlan(task_id="demo", steps=[
        PlanStep(id="research", capability=Capability.RAG),
        PlanStep(id="report", capability=Capability.DOCUMENT, depends_on=["research"]),
    ])
    executor = SovereignExecutor(PlanValidator(), AuditChain(tmp_path / "audit.jsonl"))
    result = await executor.execute(plan, executor=lambda step: {"ok": True, "status": "success", "tool": step.id})
    assert result["passed"] and result["audit_valid"]
    assert result["steps"][0]["route"]["mode"] == "deterministic"
    assert all(item["verification"]["passed"] for item in result["steps"])


@pytest.mark.asyncio
async def test_sovereign_execution_high_risk_is_blocked_without_approval(tmp_path: Path):
    plan = ExecutionPlan(steps=[PlanStep(id="deploy", capability=Capability.TOOL, tool="deploy production")])
    executor = SovereignExecutor(PlanValidator(), AuditChain(tmp_path / "audit.jsonl"))
    denied = await executor.execute(plan, executor=lambda _: {"ok": True, "status": "success"})
    assert not denied["passed"] and "approval" in denied["steps"][0]["error"]
    accepted = await executor.execute(plan, executor=lambda _: {"ok": True, "status": "success"}, approval=lambda *_: True)
    assert accepted["passed"]


@pytest.mark.asyncio
async def test_required_citations_and_deliverable_are_verified(tmp_path: Path):
    artifact = tmp_path / "report.txt"
    artifact.write_text("verified")
    plan = ExecutionPlan(steps=[PlanStep(id="deliver", capability=Capability.DOCUMENT,
        metadata={"require_citations": True, "require_deliverable": True})])
    executor = SovereignExecutor(PlanValidator(), AuditChain(tmp_path / "audit.jsonl"))
    missing = await executor.execute(plan, executor=lambda _: {"ok": True, "status": "success"})
    assert not missing["passed"]
    passed = await executor.execute(plan, executor=lambda _: {"ok": True, "status": "success",
        "deliverable": str(artifact), "citations": [{"source": "official-record"}]})
    assert passed["passed"]


@pytest.mark.asyncio
async def test_sovereign_execution_failure_recovery_checkpoints_and_retries(tmp_path: Path):
    calls = 0
    def flaky(_):
        nonlocal calls
        calls += 1
        return {"ok": calls == 2, "status": "success" if calls == 2 else "failed"}
    plan = ExecutionPlan(steps=[PlanStep(id="code", capability=Capability.CODING, metadata={"retries": 1})])
    result = await SovereignExecutor(PlanValidator(), AuditChain(tmp_path / "audit.jsonl")).execute(plan, executor=flaky)
    assert result["passed"] and result["steps"][0]["attempt"] == 2


@pytest.mark.asyncio
async def test_sovereign_execution_denied_tool_never_invokes_executor(tmp_path: Path):
    called = False
    def should_not_run(_):
        nonlocal called
        called = True
        return {"ok": True, "status": "success"}
    plan = ExecutionPlan(steps=[PlanStep(id="delete", capability=Capability.TOOL, tool="delete records")])
    result = await SovereignExecutor(PlanValidator(), AuditChain(tmp_path / "audit.jsonl")).execute(plan, executor=should_not_run)
    assert not result["passed"] and not called


@pytest.mark.asyncio
async def test_goal_budget_is_enforced_before_execution(tmp_path: Path):
    with pytest.raises(ValueError, match="step budget"):
        ExecutionPlan(
            task_id="budgeted",
            goal=GoalSpec(objective="bounded work", budget=GoalBudget(max_steps=1)),
            steps=[
                PlanStep(id="one", capability=Capability.TOOL),
                PlanStep(id="two", capability=Capability.TOOL, depends_on=["one"]),
            ],
        )

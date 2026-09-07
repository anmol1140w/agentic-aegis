import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.golden import evaluate_golden_tasks
from evaluation.routing import evaluate_baseline_vs_adaptive, reward_for_outcome
from models.registry import ModelRegistry
from runtime.agents import AgentResult, AgentStatus
from runtime.compound_tasks import decompose_task
from runtime.official_documents import OfficialDocumentWorkflow
from routing.approval_note import ApprovalNote
from runtime.task_graph import run_task_graph
from runtime.tool_policy import PolicyDenied, PolicyEngine
from routing.adaptive_router import BanditDecision, ContextualBanditRouter, EvaluationGate, RoutingCandidate, SQLiteBanditStore
from storage.telemetry import SQLiteTelemetryStore
from tools.calculator import calculator
from tools.registry import ToolRegistry


def test_golden_evaluation_is_repeatable_and_perfect() -> None:
    report = evaluate_golden_tasks(repeats=3)
    assert report["task_count"] == 24
    assert report["evaluations"] == 72
    assert report["accuracy"] == 1.0


def test_compound_task_has_ordered_stages() -> None:
    stages = decompose_task("Read this scanned inspection report and calculate remaining life for a valve and draft an approval note")
    assert [stage.name for stage in stages] == ["extract", "interpret_visual", "calculate", "draft_approval", "verify"]


def test_official_document_requires_approval_then_generates(tmp_path: Path) -> None:
    workflow = OfficialDocumentWorkflow(tmp_path / "deliverables")
    note = ApprovalNote(subject="Valve replacement", background="Damaged valve", proposal="Replace it",
                        financial_implication="INR 1", dop_authority="Plant Head", recommendation="Approve")
    state = workflow.prepare(note, request_id="doc-test")
    denied = workflow.generate(note, request_id=state.request_id)
    assert denied["status"] == "approval_required"
    generated = workflow.generate(note, request_id=state.request_id, approve=True)
    assert generated["status"] == "generated"
    assert Path(generated["path"]).is_file()


@pytest.mark.asyncio
async def test_compound_graph_executes_all_stages() -> None:
    class Specialist:
        descriptor = SimpleNamespace(provider_name="local", name="document_agent")
        async def run(self, request):
            return AgentResult(agent="document_agent", status=AgentStatus.SUCCESS,
                               summary="stage complete", verification={"required": True, "status": "passed"})
    class Registry:
        def get(self, name): return Specialist()
    class Master:
        registry = Registry()
        def _capability_plan(self, request):
            return [{"agent": "document_agent", "capability": "compound", "selection_score": 1.0}]
        async def delegate_to_agent(self, *args, **kwargs): return await Specialist().run(None)
    state = await run_task_graph(Master(), "Read this scanned inspection report and calculate remaining life", workspace_root=".")
    assert state["status"] == "completed"
    assert len(state["plan"]) == 4
    assert len(state["step_results"]) == 4


def test_telemetry_persists_structured_execution(tmp_path: Path) -> None:
    store = SQLiteTelemetryStore(tmp_path / "telemetry.db")
    record = store.record(run_id="run-1", task_type="inspection_analysis",
                          required_capabilities=["vision", "reasoning"], selected_model="qwen-general",
                          latency_ms=42.5, tool_calls=2, verification_result=True,
                          human_approval=True, final_success=True)
    assert record["final_success"] is True
    assert store.list_recent(1)[0]["required_capabilities"] == ["vision", "reasoning"]
    store.close()


def test_linucb_shadow_and_promotion_gate_are_security_independent() -> None:
    baseline = RoutingCandidate("document", "qwen-general", frozenset({"document_analysis"}), 1.0, "document")
    alternate = RoutingCandidate("vision", "qwen-vision", frozenset({"document_analysis"}), 1.0, "vision")
    router = ContextualBanditRouter(alpha=0.1)
    decision = router.select({"task_type": "document", "quality_required": 0.8}, [baseline, alternate], baseline)
    assert decision.mode == "shadow"
    assert decision.selected == baseline
    good = {"task_success_rate": 1.0, "verification_pass_rate": 1.0, "schema_evidence_validity": 1.0,
            "average_latency_ms": 10, "timeout_rate": 0, "coverage": 1.0,
            "actual_workflow_execution": True, "executed_candidate_observations": 2}
    result = EvaluationGate().compare(good, {**good, "security_violations": 1})
    assert not result.promoted
    assert "security violations detected" in result.reasons


def test_baseline_adaptive_routing_evaluation_promotes_only_non_inferior_policy() -> None:
    report = evaluate_baseline_vs_adaptive(repeats=3)
    assert report["baseline"]["task_count"] == 24
    assert report["adaptive"]["task_success_rate"] >= report["baseline"]["task_success_rate"]
    assert report["promotion"]["promoted"] is False
    assert "actual candidate workflow execution evidence is unavailable" in report["promotion"]["reasons"]
    assert reward_for_outcome(success=True, verification=True, latency_ms=0) == 0.95
    assert -1.0 <= reward_for_outcome(success=False, verification=False, latency_ms=30_000,
                                       failures=1, timed_out=True) <= 1.0
    assert reward_for_outcome(success=True, verification=True, latency_ms=0,
                              security_violation=True) == -1.0


def test_linucb_learns_preferred_action_from_observed_rewards_and_persists(tmp_path: Path) -> None:
    coder = RoutingCandidate("coder", "qwen-coder", frozenset({"coding"}), 1.0, "coding")
    reasoner = RoutingCandidate("reasoner", "qwen-general", frozenset({"coding"}), 1.0, "reasoning")
    context = {"task_type": "coding", "quality_required": 0.8}
    store = SQLiteBanditStore(tmp_path / "bandit.db")
    router = ContextualBanditRouter(alpha=0.0, min_confidence=0.0, store=store)
    router.enabled = True
    router.mode = "adaptive"
    for _ in range(20):
        for candidate, reward in ((coder, 1.0), (reasoner, -1.0)):
            decision = BanditDecision(candidate, candidate, (coder, reasoner), "adaptive")
            router.record_outcome(decision, context, reward)
    decision = router.select(context, [coder, reasoner], reasoner, mode="adaptive")
    assert decision.selected.name == "coder"
    assert router.diagnostics()["training_observations"] == 40
    store.close()
    reloaded_store = SQLiteBanditStore(tmp_path / "bandit.db")
    reloaded = ContextualBanditRouter(alpha=0.0, min_confidence=0.0, store=reloaded_store)
    reloaded.enabled = True
    assert reloaded.select(context, [coder, reasoner], reasoner, mode="adaptive").selected.name == "coder"
    reloaded_store.close()


def test_bandit_cold_start_and_unsafe_candidates_fall_back() -> None:
    safe = RoutingCandidate("safe", "qwen-general", frozenset({"reasoning"}), 1.0, "reasoning")
    unsafe = RoutingCandidate("network", "remote", frozenset({"reasoning"}), 1.0, "reasoning", network=True)
    router = ContextualBanditRouter(alpha=0.35, min_confidence=0.05)
    router.enabled = True
    decision = router.select({"task_type": "new", "quality_required": 0.8}, [safe, unsafe], safe, mode="adaptive")
    assert [candidate.name for candidate in decision.eligible_candidates] == ["safe"]
    assert decision.selected.name == "safe"
    empty = router.select({"task_type": "new", "quality_required": 1.1}, [safe], safe, mode="adaptive")
    assert empty.fallback is True


def test_invalid_observation_blacklists_action_for_context(tmp_path: Path) -> None:
    safe = RoutingCandidate("safe", "qwen-general", frozenset({"reasoning"}), 1.0, "reasoning")
    alternate = RoutingCandidate("alternate", "qwen-general", frozenset({"reasoning"}), 1.0, "reasoning")
    context = {"task_type": "security-sensitive", "quality_required": 0.8}
    store = SQLiteBanditStore(tmp_path / "invalid.db")
    router = ContextualBanditRouter(alpha=0.0, min_confidence=0.0, store=store)
    router.enabled = True
    router.mode = "adaptive"
    decision = router.select(context, [safe, alternate], safe, mode="adaptive")
    router.record_outcome(decision, context, -1.0, invalid=True)
    after = router.select(context, [safe, alternate], safe, mode="adaptive")
    assert decision.selected.name not in {candidate.name for candidate in after.eligible_candidates}
    store.close()


def test_subprocess_crash_restart_resumes_from_durable_checkpoint(tmp_path: Path) -> None:
    """Acceptance test: a killed worker resumes the same run from SQLite."""
    db = tmp_path / "runs.db"
    worker = r'''
import asyncio, json, os, sys
from types import SimpleNamespace
from runtime.agents import AgentResult, AgentStatus
from runtime.task_graph import run_task_graph
from storage.graph_state import SQLiteGraphStateStore

class Specialist:
    descriptor = SimpleNamespace(provider_name="local", name="document_agent")
    async def run(self, request):
        return AgentResult(agent="document_agent", status=AgentStatus.SUCCESS,
                           summary="stage evidence", verification={"required": True, "status": "passed"})

class Registry:
    def get(self, name): return Specialist()

class Master:
    registry = Registry()
    async def route_request(self, request, context=None):
        return [{"agent": "document_agent", "capability": "compound", "selection_score": 1.0}]
    async def delegate_to_agent(self, *args, **kwargs): return await Specialist().run(None)

async def main():
    store = SQLiteGraphStateStore(sys.argv[1])
    def progress(event):
        if os.getenv("AEGIS_CRASH_AFTER") == event.get("node") == "execute_step":
            os._exit(23)
    state = await run_task_graph(Master(),
        "Read this scanned inspection report, calculate remaining life, and draft an approval note",
        workspace_root=sys.argv[2], state_store=store, run_id="run-crash-resume",
        progress_callback=progress)
    print(json.dumps({"status": state.get("status"), "steps": len(state.get("step_results", [])),
                      "node": state.get("current_step_index")}))

asyncio.run(main())
'''
    env = {**os.environ, "PYTHONPATH": str(Path.cwd())}
    first_env = {**env, "AEGIS_CRASH_AFTER": "execute_step"}
    first = subprocess.run([sys.executable, "-c", worker, str(db), str(tmp_path)],
                           cwd=Path.cwd(), env=first_env, capture_output=True, text=True)
    assert first.returncode == 23, first.stderr
    second_env = {key: value for key, value in env.items() if key != "AEGIS_CRASH_AFTER"}
    second = subprocess.run([sys.executable, "-c", worker, str(db), str(tmp_path)],
                            cwd=Path.cwd(), env=second_env, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    resumed = json.loads(second.stdout.strip().splitlines()[-1])
    assert resumed["status"] == "completed"
    assert resumed["steps"] >= 5
    assert "execute_step" in first.stdout or db.exists()

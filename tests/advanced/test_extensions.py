import pytest

from aegis.contracts import Capability, ExecutionPlan, PlanStep
from aegis.sovereign import (A2AGateway, A2AMessage, DSPyProgram, FederatedInference,
                             KubernetesPlanner, RLRoutingStrategy, TEEVerifier, VisualWorkflowEditor)
from routing.adaptive_router import ContextualBanditRouter


@pytest.mark.asyncio
async def test_a2a_transport_extension():
    gateway = A2AGateway()
    gateway.register("reviewer", lambda message: {"ok": True, "trace": message.trace_id})
    assert (await gateway.dispatch(A2AMessage("t", "master", "reviewer", "review", {}, "trace")))["ok"]


@pytest.mark.asyncio
async def test_federated_inference_extension():
    result = await FederatedInference().infer("private prompt", [lambda _: {"ok": True, "answer": "a"}])
    assert result["ok"] and result["worker_count"] == 1 and "prompt_sha256" in result


def test_dspy_tee_visual_and_kubernetes_extension_contracts():
    assert DSPyProgram("task -> answer").run({"task": "hello"})["ok"]
    assert TEEVerifier({"known"}).verify("worker", "known", "nonce").verified
    plan = ExecutionPlan(steps=[PlanStep(id="one", capability=Capability.REASONING)])
    assert VisualWorkflowEditor.export(plan)["nodes"][0]["id"] == "one"
    manifest = KubernetesPlanner.manifest("worker", "aegis:local", tee_required=True)
    assert manifest["metadata"]["annotations"]["aegis.io/tee-required"] == "true"


def test_rl_strategy_is_experimental_and_falls_back_while_disabled():
    plan = ExecutionPlan(steps=[PlanStep(id="route", capability=Capability.RAG)])
    route = RLRoutingStrategy(ContextualBanditRouter()).route(plan.steps[0])
    assert route["mode"] == "deterministic" and route["rl"] == "shadow/disabled"

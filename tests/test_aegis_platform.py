from pathlib import Path

import pytest

from aegis.contracts import Capability, ExecutionPlan, PlanStep, PlanValidator
from aegis.governance import AuditChain, PolicyEngine, Principal
from aegis.memory import MemoryStore
from aegis.retrieval import HybridKnowledgeStore


def test_typed_plan_validation_catches_capability_and_cycle():
    plan = ExecutionPlan(steps=[
        PlanStep(id="a", capability=Capability.CODING, model="small", depends_on=["b"]),
        PlanStep(id="b", capability=Capability.RAG, model="small", depends_on=["a"]),
    ])
    errors = PlanValidator(model_capabilities={"small": {"rag"}}).validate(plan)
    assert any("cycle" in error for error in errors)
    assert any("lacks coding" in error for error in errors)


def test_rbac_policy_and_hash_chain(tmp_path: Path):
    user = Principal("u", frozenset({"user"}), department="finance")
    with pytest.raises(PermissionError):
        PolicyEngine().check(user, "tool:use")
    chain = AuditChain(tmp_path / "audit.jsonl")
    chain.append("task_created", actor="u", task_id="t1")
    chain.append("task_finished", actor="u", task_id="t1")
    assert chain.verify()


@pytest.mark.asyncio
async def test_hybrid_retrieval_returns_citations():
    store = HybridKnowledgeStore()
    store.add("MCP tools are brokered through the approved gateway.", "security.md")
    result = await store.search("approved MCP gateway")
    assert result and result[0].source == "security.md"


def test_memory_is_scoped_and_searchable(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.put("routing prefers local models", scope="router")
    store.put("unrelated note", scope="other")
    assert len(store.search("local models", scope="router")) == 1
    assert store.search("local models", scope="other") == []

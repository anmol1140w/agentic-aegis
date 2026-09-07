"""Golden routing tasks for the MRPL SIH domain boundary."""

from routing.approval_note import APPROVAL_NOTE_FIELDS, ApprovalNote, explain_approval_note
from routing.classifier import DomainIntent, TaskClassifier, TaskType
from runtime.task_graph import run_task_graph
from runtime.agents import AgentResult, AgentStatus
from types import SimpleNamespace
import pytest


class _ApprovalNoteMaster:
    class _Registry:
        def get(self, name):
            return SimpleNamespace(descriptor=SimpleNamespace(provider_name="local"))
    registry = _Registry()

    def _capability_plan(self, request):
        return [{"agent": "document_agent", "capability": "psu_approval_note", "selection_score": 1.0}]

    async def delegate_to_agent(self, *args, **kwargs):
        return AgentResult(agent="document_agent", status=AgentStatus.SUCCESS, summary="unused")


def test_refinery_approval_note_means_psu_administrative_note() -> None:
    task = TaskClassifier().classify("What is a refinery approval note?")
    assert task.domain_intent is DomainIntent.PSU_APPROVAL_NOTE
    assert task.workflow == "psu_approval_note_explain"
    assert "product_operations" not in task.interpretation_candidates
    assert "financial implication" in explain_approval_note()


def test_draft_approval_note_selects_generation_workflow() -> None:
    task = TaskClassifier().classify("Draft an approval note for replacing a damaged pipeline valve.")
    assert task.domain_intent is DomainIntent.PSU_APPROVAL_NOTE
    assert task.workflow == "psu_approval_note_generate"
    assert task.requires_human_approval is True
    assert set(APPROVAL_NOTE_FIELDS) == set(ApprovalNote.model_fields)


def test_diesel_shipment_is_product_operations() -> None:
    task = TaskClassifier().classify("How much diesel should be shipped?")
    assert task.domain_intent is DomainIntent.PRODUCT_OPERATIONS
    assert task.workflow == "product_operations"


def test_sih_compound_workflows_have_explicit_domain_signals() -> None:
    classifier = TaskClassifier()
    cases = {
        "Summarize this PDF": TaskType.SUMMARIZATION,
        "Read this scanned inspection report and calculate remaining life": TaskType.DOCUMENT_ANALYSIS,
        "Read this P&ID and identify valve tags": TaskType.ENGINEERING_DOCUMENT,
        "Write Python code to analyze this Excel file": TaskType.CODING,
    }
    for prompt, expected_type in cases.items():
        task = classifier.classify(prompt)
        assert task.task_type is expected_type, prompt
        assert task.required_capabilities, prompt
    scanned = classifier.classify("Read this scanned inspection report and calculate remaining life")
    assert {"ocr", "calculation"}.issubset(set(scanned.required_capabilities))
    assert "vision" not in scanned.required_capabilities


@pytest.mark.asyncio
async def test_approval_note_explain_is_grounded_before_model_generation() -> None:
    state = await run_task_graph(_ApprovalNoteMaster(), "What is a refinery approval note?", workspace_root=".")
    assert state["status"] == "completed"
    assert "Delegation of Power" in state["final_answer"]
    assert "shipment or transaction" in state["final_answer"]

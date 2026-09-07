from types import SimpleNamespace

import pytest

from runtime.agents import AgentResult, AgentStatus
from runtime.task_graph import run_task_graph


class _Specialist:
    def __init__(self, result):
        self.descriptor = SimpleNamespace(provider_name="local", name="document_agent")
        self.result = result

    async def run(self, request):
        return self.result


class _Registry:
    def __init__(self, specialist):
        self.specialist = specialist

    def get(self, name):
        return self.specialist


class _Master:
    def __init__(self, result):
        self.registry = _Registry(_Specialist(result))

    def _capability_plan(self, request):
        return [{"agent": "document_agent", "capability": "document_creation", "selection_score": 0.9}]

    async def delegate_to_agent(self, agent, task, **kwargs):
        return await self.registry.get(agent).run(None)


class _RepairMaster:
    def __init__(self):
        self.registry = _Registry(_Specialist(AgentResult(agent="document_agent", status=AgentStatus.SUCCESS)))
        self.contexts = []
        self.calls = 0

    def _capability_plan(self, request):
        return [{"agent": "document_agent", "capability": "document_creation", "selection_score": 0.9}]

    async def delegate_to_agent(self, agent, task, **kwargs):
        self.contexts.append(kwargs.get("context", {}))
        self.calls += 1
        if self.calls == 1:
            result = AgentResult(
                agent=agent, status=AgentStatus.FAILURE, summary="pytest failed",
                errors=["assertion failed"],
                metadata={"coding_state": {
                    "files_read": {"src/app.py": "return 1"},
                    "last_tool_result": {"tool": "execute_command", "exit_code": 1,
                                          "stdout": "1 failed", "stderr": "assertion failed"},
                    "command_executed": True, "verification": {"status": "failed"},
                    "evidence": ["failure:execute_command:assertion failed"],
                    "state_version": 0,
                }})
            return result
        return AgentResult(agent=agent, status=AgentStatus.SUCCESS, summary="repaired",
                           verification={"required": True, "status": "passed"})


class _CodingSuccessWithoutEvidence(_Master):
    def __init__(self, result):
        super().__init__(result)

    def _capability_plan(self, request):
        return [{"agent": "coding_agent", "capability": "code_debugging"}]


class _CodingRegistry(_Registry):
    def get(self, name):
        specialist = self.specialist
        specialist.descriptor.name = name
        return specialist


@pytest.mark.asyncio
async def test_universal_graph_verifies_document_result():
    result = AgentResult(agent="document_agent", status=AgentStatus.SUCCESS,
                         summary="created", verification={"required": True, "status": "passed"})
    state = await run_task_graph(_Master(result), "create txt about AEGIS", workspace_root=".")
    assert state["status"] == "completed"
    assert state["verification"]["passed"] is True
    assert any(e.get("event") == "verification_passed" for e in state["trace"])


@pytest.mark.asyncio
async def test_universal_graph_has_bounded_structured_repair():
    result = AgentResult(agent="document_agent", status=AgentStatus.FAILURE,
                         summary="writer unavailable", errors=["writer_unavailable"])
    state = await run_task_graph(_Master(result), "create pdf about AEGIS", workspace_root=".", max_task_retries=1)
    assert state["status"] == "failed"
    assert state["repair_history"] == [{"action": "retry", "tool": None, "parameters": {},
                                        "reason": "retryable specialist failure"}]
    assert state["errors"]


@pytest.mark.asyncio
async def test_universal_graph_does_not_retry_policy_denial():
    result = AgentResult(agent="document_agent", status=AgentStatus.FAILURE,
                         summary="denied", errors=["approval_denied"])
    state = await run_task_graph(_Master(result), "create pdf about AEGIS", workspace_root=".", max_task_retries=3)
    assert state["status"] == "failed"
    assert state["repair_history"] == []
    assert any(item.get("error_code") == "approval_denied" for item in state["errors"])


@pytest.mark.asyncio
async def test_max_tool_steps_cannot_be_reported_as_verified_completion():
    result = AgentResult(
        agent="coding_agent", status=AgentStatus.FAILURE,
        summary="Coding tool loop limit reached",
        errors=["max tool steps reached"],
        verification={"required": True, "status": "passed"},
    )
    state = await run_task_graph(_Master(result), "fix the error in fibonacci.py",
                                 workspace_root=".", max_task_retries=3)
    assert state["status"] == "failed"
    assert state["verification"]["passed"] is False
    assert any("max tool steps reached" in str(item).lower() for item in state["errors"])


@pytest.mark.asyncio
async def test_graph_repair_reuses_specialist_working_memory():
    master = _RepairMaster()
    state = await run_task_graph(master, "fix the failing test", workspace_root=".", max_task_retries=1)
    assert state["status"] == "completed"
    assert len(master.contexts) == 2
    resumed = master.contexts[1]["agent_state"]
    assert resumed["files_read"]["src/app.py"] == "return 1"
    assert resumed["last_tool_result"]["exit_code"] == 1
    assert resumed["verification"]["status"] == "failed"


@pytest.mark.asyncio
async def test_coding_success_without_execution_evidence_cannot_verify():
    result = AgentResult(agent="coding_agent", status=AgentStatus.SUCCESS,
                         summary="pytest passed", verification={"status": "passed"})
    master = _CodingSuccessWithoutEvidence(result)
    master.registry = _CodingRegistry(_Specialist(result))
    state = await run_task_graph(master, "fix the failing tests and run pytest", workspace_root=".", max_task_retries=0)
    assert state["status"] == "failed"
    assert state["verification"]["passed"] is False


@pytest.mark.asyncio
async def test_verified_file_change_overrides_contradictory_model_summary():
    result = AgentResult(
        agent="coding_agent", status=AgentStatus.SUCCESS,
        summary="Insufficient evidence to proceed.",
        changes=["doubly_linked_list.py"],
        verification={"required": True, "status": "verified"},
        metadata={"coding_state": {"verification": {"status": "verified"}, "changes": ["doubly_linked_list.py"], "evidence": ["repository_context:19 files under workspace", "syntax_validation:doubly_linked_list.py:passed"]}},
    )
    state = await run_task_graph(_CodingSuccessWithoutEvidence(result), "create a py file on doubly linked list", workspace_root=".", max_task_retries=0)
    assert state["status"] == "completed"
    assert state["final_answer"] == "Created and verified: doubly_linked_list.py"


@pytest.mark.asyncio
async def test_review_and_finish_sanitizes_raw_json_summary():
    """If the agent summary is a raw task-spec JSON blob, the final answer
    must be a human-readable string, not the raw dict stringified."""
    import json

    raw_blob = json.dumps({
        "operation": "analyze",
        "original_request": "fix the error in fibonacci.py",
        "normalized_request": "fix the error in fibonacci.py",
        "agent": "coding_agent",
        "domain_intent": "general",
        "workflow": "general_reasoning",
        "artifact_format": None,
        "modality": "text",
        "required_capabilities": ["reasoning"],
        "compound_steps": [],
    })

    # The verify_coding_result check for "fix" tasks requires command evidence.
    # Supply a successful py_compile run in commands so the coding check passes.
    result = AgentResult(
        agent="coding_agent", status=AgentStatus.SUCCESS,
        summary=raw_blob,
        changes=["fibonacci.py"],
        evidence=["edit_file:fibonacci.py (fixed)", "py_compile:success"],
        verification={"required": True, "status": "verified"},
        metadata={
            "coding_state": {
                "files_read": {"fibonacci.py": "def fibonacci(n): ..."},
                "changes": ["fibonacci.py"],
                "evidence": ["edit_file:fibonacci.py (fixed)", "py_compile:success"],
                "last_tool_result": {"tool": "execute_command", "ok": True, "exit_code": 0,
                                     "stdout": "", "stderr": ""},
                "command_executed": True,
                "verification": {"required": True, "status": "verified",
                                 "command": "python -m py_compile fibonacci.py"},
                "commands": [{
                    "command": "python -m py_compile fibonacci.py",
                    "exit_code": 0, "stdout": "", "stderr": "",
                    "timed_out": False, "state_version": 1,
                }],
                "state_version": 1,
                "last_edit_state_version": 0,
            }
        },
    )

    class _SpecMaster(_Master):
        def __init__(self, r):
            self.registry = _CodingRegistry(_Specialist(r))

        def _capability_plan(self, request):
            return [{"agent": "coding_agent", "capability": "code_debugging"}]

    state = await run_task_graph(_SpecMaster(result), "fix the error in fibonacci.py",
                                 workspace_root=".", max_task_retries=0)
    final = state.get("final_answer", "")
    # Must not be the raw JSON blob
    assert "operation" not in final or "{" not in final[:5], \
        f"final_answer still contains raw JSON: {final[:200]}"
    # Must mention the changed file or a change-related keyword
    assert any(kw in final.lower() for kw in ("fibonacci", "modified", "created", "completed", "produced")), \
        f"final_answer should reference the file change: {final}"

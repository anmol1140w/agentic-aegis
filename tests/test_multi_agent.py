from __future__ import annotations

import pytest

from runtime.agents import (
    AgentCapability,
    AgentCapabilityPolicy,
    AgentDescriptor,
    AgentRegistry,
    AgentRequest,
    AgentResult,
    AgentStatus,
    BaseAgent,
    MasterAgent,
    OllamaSpecialistAgent,
)


class StubAgent(BaseAgent):
    def __init__(self, name: str, capabilities: list[AgentCapability], result: AgentResult):
        self.descriptor = AgentDescriptor(
            name=name, role=f"{name} role", capabilities=capabilities,
            provider_name="stub", allowed_tools=["read_file"],
        )
        self.result = result
        self.requests: list[AgentRequest] = []

    async def run(self, request: AgentRequest) -> AgentResult:
        self.requests.append(request)
        return self.result


def test_registry_selects_by_capability_without_model_logic() -> None:
    registry = AgentRegistry()
    agent = StubAgent("coder", [AgentCapability.CODING], AgentResult(status=AgentStatus.SUCCESS))
    registry.register(agent)
    assert registry.get("coder") is agent
    assert registry.eligible([AgentCapability.CODING])[0].name == "coder"
    assert registry.capabilities() == {"coder": ["coding"]}


@pytest.mark.asyncio
async def test_master_delegates_structured_result_and_verifies() -> None:
    registry = AgentRegistry()
    agent = StubAgent("coder", [AgentCapability.CODING], AgentResult(
        status=AgentStatus.SUCCESS, summary="implemented", artifacts=["artifacts/change.json"]
    ))
    registry.register(agent)
    trace: list[dict] = []

    async def planner(request: str, capabilities: dict[str, list[str]]) -> list[dict]:
        assert "coder" in capabilities
        return [{"agent": "coder", "task": "inspect code", "success_criteria": ["evidence"]}]

    async def verifier(result: AgentResult) -> dict:
        return {"status": "passed", "evidence": result.artifacts}

    state = await MasterAgent(registry, planner=planner, verifier=verifier, trace=trace).run("fix bug")
    assert state.final_answer == "implemented"
    assert state.agent_results[0].status == AgentStatus.SUCCESS
    assert state.verification["status"] == "passed"
    assert {event["event"] for event in trace} >= {"MASTER_PLAN", "MASTER_DELEGATE", "AGENT_COMPLETE", "MASTER_REVIEW", "VERIFICATION", "FINAL"}


@pytest.mark.asyncio
async def test_master_recovers_from_failed_agent_and_is_bounded() -> None:
    registry = AgentRegistry()
    failed = StubAgent("bad", [AgentCapability.GENERAL], AgentResult(status=AgentStatus.FAILURE, errors=["timeout"]))
    good = StubAgent("good", [AgentCapability.GENERAL], AgentResult(status=AgentStatus.SUCCESS, summary="recovered"))
    registry.register(failed); registry.register(good)
    state = await MasterAgent(registry, planner=lambda *_: _plan(), max_subagent_calls=2).run("task")
    assert state.final_answer == "recovered"
    assert state.errors == ["timeout"]


@pytest.mark.asyncio
async def test_document_failure_never_falls_back_to_coding_agent() -> None:
    registry = AgentRegistry()
    failed_doc = StubAgent("document_agent", [AgentCapability.DOCUMENT], AgentResult(status=AgentStatus.FAILURE, summary="TXT writer failed", errors=["artifact_creation_failed"]))
    coder = StubAgent("coding_agent", [AgentCapability.CODING], AgentResult(status=AgentStatus.SUCCESS, summary="wrong fallback"))
    registry.register(failed_doc); registry.register(coder)

    async def planner(*_):
        return [{"agent": "document_agent", "task": "create txt about Agentic AI"},
                {"agent": "coding_agent", "task": "create txt about Agentic AI"}]

    trace: list[dict] = []
    state = await MasterAgent(registry, planner=planner, trace=trace).run("create txt about Agentic AI")
    assert len(failed_doc.requests) == 1
    assert not coder.requests
    assert state.agent_results[0].status == AgentStatus.FAILURE
    assert any(e["event"] == "MASTER_REPLAN" and e["required_capability"] == "document_agent" for e in trace)


@pytest.mark.asyncio
async def test_coding_file_generation_requires_verified_change() -> None:
    class FinalProvider:
        async def generate(self, _prompt):
            from models.base import ModelResponse
            return ModelResponse(content='{"action":"final","answer":"created"}', model="fake")
    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="coding_agent", role="coding", capabilities=[AgentCapability.CODING], provider_name="stub", allowed_tools=["read_file", "edit_file"]),
        FinalProvider(), {},
    )
    result = await agent.run(AgentRequest(task="create a Python program and write it to the workspace"))
    assert result.status == AgentStatus.FAILURE
    assert "file_generation_unverified" in result.errors


async def _plan() -> list[dict]:
    return [{"agent": "bad", "task": "try"}, {"agent": "good", "task": "retry"}]


def test_policy_is_least_privilege_and_approval_gated() -> None:
    policy = AgentCapabilityPolicy()
    descriptor = AgentDescriptor(name="doc", role="document", capabilities=[AgentCapability.DOCUMENT], provider_name="stub", allowed_tools=["ocr_pdf"])
    assert policy.can_use_tool(descriptor, "ocr_pdf")
    assert not policy.can_use_tool(descriptor, "read_file")
    assert policy.authorize("read", "t1")
    assert not policy.authorize("network", "t1")
    assert not policy.authorize("edit", "t1")
    assert policy.authorize("edit", "t1", approve=lambda *_: True)


def test_repository_analysis_routes_to_coding_workspace_agent() -> None:
    registry = AgentRegistry()
    master = MasterAgent(registry)
    plan = master._capability_plan("Analyze this repository and explain its architecture")
    assert plan[0]["agent"] == "coding_agent"
    assert plan[0]["capability"] == "codebase_understanding"


def test_terminal_python_command_routes_to_coding_workspace_agent() -> None:
    registry = AgentRegistry()
    master = MasterAgent(registry)
    plan = master._capability_plan('run this python command: python -c "print(2 + 2)"')
    assert plan[0]["agent"] == "coding_agent"
    assert plan[0]["capability"] == "terminal_execution"


def test_source_path_and_test_repair_route_to_coding_agent() -> None:
    master = MasterAgent(AgentRegistry())
    plan = master._capability_plan(
        "Fix the tests for workspace/agent_test/calculator.py and run pytest"
    )
    assert plan[0]["agent"] == "coding_agent"


@pytest.mark.asyncio
async def test_delegate_to_agent_returns_structured_result_and_bounds_depth() -> None:
    registry = AgentRegistry()
    agent = StubAgent("coder", [AgentCapability.CODING], AgentResult(status=AgentStatus.SUCCESS, summary="ok"))
    registry.register(agent)
    master = MasterAgent(registry, max_depth=1)
    result = await master.delegate_to_agent("coder", "inspect", depth=0)
    assert result.status == AgentStatus.SUCCESS
    blocked = await master.delegate_to_agent("coder", "nested", depth=2)
    assert blocked.status == AgentStatus.BLOCKED


@pytest.mark.asyncio
async def test_coding_agent_reads_source_before_analysis() -> None:
    from unittest.mock import AsyncMock, MagicMock

    source = {"ok": True, "tool": "read_file", "path": "verify_runs.py", "content": "assert True\n"}
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=MagicMock(content='{"status":"success","summary":"Potential issue reviewed"}'))
    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="coding_agent", role="coding", capabilities=[AgentCapability.CODING], provider_name="stub", allowed_tools=["read_file"]),
        provider,
        tools={"read_file": lambda path: source},
    )
    result = await agent.run(AgentRequest(task="Inspect verify_runs.py and identify any potential bug. Do not modify files."))
    assert result.status == AgentStatus.SUCCESS
    assert any("verify_runs.py" in item for item in result.evidence)
    assert "verify_runs.py" in result.artifacts
    assert not any(tool in provider.generate.call_args.args[0] for tool in ("edit_file", "execute_command"))


@pytest.mark.asyncio
async def test_coding_agent_allows_new_file_creation_without_initial_read(tmp_path) -> None:
    from unittest.mock import AsyncMock, MagicMock

    calls: list[str] = []
    source = "def factorial(n):\n    return 1\n"
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=[
        MagicMock(content='{"action":"tool","tool":"create_file","arguments":{"path":"factorial.py","content":"def factorial(n):\\n    return 1\\n"}}'),
        MagicMock(content='{"action":"final","answer":"created and verified"}'),
    ])

    def read_file(path):
        calls.append("read")
        return {"ok": True, "content": source, "path": path}

    def create_file(path, content):
        calls.append("create")
        (tmp_path / path).write_text(content, encoding="utf-8")
        return {"ok": True, "path": path}

    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="coding_agent", role="coding", capabilities=[AgentCapability.CODING],
                        provider_name="stub", allowed_tools=["read_file", "create_file"]),
        provider,
        tools={"read_file": read_file, "create_file": create_file},
    )
    result = await agent.run(AgentRequest(
        task="Create a Python file called factorial.py in the workspace.",
        context={"workspace_root": str(tmp_path), "path": "factorial.py"},
    ))
    assert result.status == AgentStatus.SUCCESS
    assert calls == ["create", "read"]


@pytest.mark.asyncio
async def test_coding_agent_must_run_requested_file_before_finalizing(tmp_path) -> None:
    """A premature model final cannot skip the user's explicit run request."""
    from unittest.mock import AsyncMock, MagicMock

    source = tmp_path / "random_proxy.py"
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=[
        MagicMock(content='{"action":"create_file","arguments":{"path":"random_proxy.py","content":"print(42)\\n"}}'),
        # This is the regression: the model tries to summarize after writing.
        MagicMock(content='{"action":"final","answer":"The random proxy was created."}'),
        MagicMock(content='{"action":"execute_command","arguments":{"command":"python random_proxy.py"}}'),
        MagicMock(content='{"action":"final","answer":"Created and ran the file."}'),
    ])

    def create_file(path, content):
        source.write_text(content, encoding="utf-8")
        return {"ok": True, "status": "success", "path": path}

    def read_file(path):
        return {"ok": True, "status": "success", "path": path, "content": source.read_text(encoding="utf-8")}

    def execute_command(**_):
        return {"ok": True, "status": "success", "exit_code": 0, "stdout": "42\n", "stderr": ""}

    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="coding_agent", role="coding", capabilities=[AgentCapability.CODING],
                        provider_name="stub", allowed_tools=["read_file", "create_file", "execute_command"]),
        provider,
        tools={"read_file": read_file, "create_file": create_file, "execute_command": execute_command},
    )
    result = await agent.run(AgentRequest(task="create proxy python file and run it and show me the output"))

    assert result.status == AgentStatus.SUCCESS
    assert result.verification["status"] == "passed"
    assert result.verification["command"] == "python random_proxy.py"
    assert provider.generate.await_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_coding_mutation_reuses_approval_and_verification(approved: bool, tmp_path) -> None:
    from unittest.mock import AsyncMock, MagicMock
    from tools.workspace import WorkspaceReadTools

    root = tmp_path / "workspace"; root.mkdir()
    target = root / "math_lib.py"; target.write_text("return 1\n", encoding="utf-8")
    tools = WorkspaceReadTools(root, approver=(lambda *_: approved), command_approver=(lambda *_: approved))
    responses = iter([
        '{"action":"edit_file","arguments":{"path":"math_lib.py","old_text":"return 1","new_text":"return 2"}}',
        '{"action":"read_file","arguments":{"path":"math_lib.py"}}',
        '{"action":"execute_command","arguments":{"command":"pytest test_math.py"}}',
    ])
    provider = MagicMock(); provider.generate = AsyncMock(side_effect=[MagicMock(content=next(responses)) for _ in range(3)])
    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="coding_agent", role="coding", capabilities=[AgentCapability.CODING], provider_name="stub",
                        allowed_tools=["read_file", "edit_file", "execute_command"]),
        provider, tools={"read_file": tools.read_file, "edit_file": tools.edit_file, "execute_command": tools.execute_command},
    )
    result = await agent.run(AgentRequest(task="Fix the bug in math_lib.py and run pytest test_math.py"))
    if approved:
        assert result.status == AgentStatus.FAILURE  # test command is allowed but fixture has no test
        assert result.changes == ["math_lib.py"]
        assert result.verification["status"] == "failed"
        assert target.read_text() == "return 2\n"
    else:
        assert result.status == AgentStatus.FAILURE
        assert result.approvals[0]["status"] == "denied"
        assert target.read_text() == "return 1\n"


@pytest.mark.asyncio
async def test_coding_agent_diagnoses_failed_command_then_repairs_and_retries(tmp_path):
    """A failed test is evidence for the same bounded coding loop, not a dead end."""
    from unittest.mock import AsyncMock, MagicMock

    source = tmp_path / "math_lib.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=[
        MagicMock(content='{"action":"read_file","arguments":{"path":"math_lib.py"}}'),
        MagicMock(content='{"action":"execute_command","arguments":{"command":"pytest test_math.py"}}'),
        MagicMock(content='{"action":"read_file","arguments":{"path":"math_lib.py"}}'),
        MagicMock(content='{"action":"edit_file","arguments":{"path":"math_lib.py","old_text":"return 1","new_text":"return 2"}}'),
        MagicMock(content='{"action":"execute_command","arguments":{"command":"pytest test_math.py"}}'),
        MagicMock(content='{"action":"final","answer":"Fixed and verified after diagnosing the failing test."}'),
    ])
    command_results = iter([
        {"ok": False, "status": "failure", "tool": "execute_command", "exit_code": 1,
         "stdout": "1 failed", "stderr": "assert 1 == 2"},
        {"ok": True, "status": "success", "tool": "execute_command", "exit_code": 0,
         "stdout": "1 passed", "stderr": ""},
    ])

    def read_file(path):
        return {"ok": True, "tool": "read_file", "path": path, "content": source.read_text()}

    def edit_file(path, old_text, new_text):
        text = source.read_text().replace(old_text, new_text, 1)
        source.write_text(text)
        return {"ok": True, "status": "success", "tool": "edit_file", "path": path}

    def execute_command(**_):
        return next(command_results)

    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="coding_agent", role="coding", capabilities=[AgentCapability.CODING],
                        provider_name="stub", allowed_tools=["read_file", "edit_file", "execute_command"]),
        provider, tools={"read_file": read_file, "edit_file": edit_file, "execute_command": execute_command},
    )
    result = await agent.run(AgentRequest(task="Fix the failing test in math_lib.py and run pytest test_math.py"))

    assert result.status == AgentStatus.SUCCESS
    assert "failure:execute_command" in " ".join(result.evidence)
    assert source.read_text() == "def value():\n    return 2\n"
    assert provider.generate.await_count == 6


@pytest.mark.asyncio
async def test_duplicate_command_is_rejected_until_state_changes(tmp_path):
    from unittest.mock import AsyncMock, MagicMock

    source = tmp_path / "math_lib.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=[
        MagicMock(content='{"action":"read_file","arguments":{"path":"math_lib.py"}}'),
        MagicMock(content='{"action":"execute_command","arguments":{"command":"pytest test_math.py"}}'),
        MagicMock(content='{"action":"execute_command","arguments":{"command":"pytest test_math.py"}}'),
        MagicMock(content='{"action":"edit_file","arguments":{"path":"math_lib.py","old_text":"return 1","new_text":"return 2"}}'),
        MagicMock(content='{"action":"execute_command","arguments":{"command":"pytest test_math.py"}}'),
        MagicMock(content='{"action":"final","answer":"Fixed after duplicate command recovery."}'),
    ])
    command_results = iter([
        {"ok": False, "status": "failure", "tool": "execute_command", "exit_code": 1, "stderr": "assertion failed", "stdout": ""},
        {"ok": True, "status": "success", "tool": "execute_command", "exit_code": 0, "stderr": "", "stdout": "1 passed"},
    ])

    def read_file(path):
        return {"ok": True, "status": "success", "path": path, "content": source.read_text()}

    def edit_file(path, old_text, new_text):
        source.write_text(source.read_text().replace(old_text, new_text, 1))
        return {"ok": True, "status": "success", "path": path}

    def execute_command(**_):
        return next(command_results)

    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="coding_agent", role="coding", capabilities=[AgentCapability.CODING],
                        provider_name="stub", allowed_tools=["read_file", "edit_file", "execute_command"]),
        provider, tools={"read_file": read_file, "edit_file": edit_file, "execute_command": execute_command},
    )
    result = await agent.run(AgentRequest(task="Fix math_lib.py and run pytest test_math.py"))

    assert result.status == AgentStatus.SUCCESS
    assert any("duplicate_action:execute_command" in item for item in result.evidence)
    assert source.read_text().endswith("return 2\n")


@pytest.mark.asyncio
async def test_document_specialist_wraps_existing_pipeline() -> None:
    from unittest.mock import MagicMock
    provider = MagicMock()
    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="document_agent", role="document", capabilities=[AgentCapability.DOCUMENT], provider_name="stub"),
        provider, tools={"document_runner": lambda path: {"status": "complete", "artifacts": ["findings.json"]}},
    )
    result = await agent.run(AgentRequest(task="inspect", context={"input_path": "report.pdf"}))
    assert result.status == AgentStatus.SUCCESS
    assert result.artifacts == ["findings.json"]


@pytest.mark.asyncio
async def test_vision_specialist_returns_structured_failure() -> None:
    from unittest.mock import AsyncMock, MagicMock
    provider = MagicMock()
    provider.encode_images.return_value = []
    provider.chat = AsyncMock(side_effect=RuntimeError("unavailable"))
    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="vision_agent", role="vision", capabilities=[AgentCapability.VISION], provider_name="stub"),
        provider,
    )
    result = await agent.run(AgentRequest(task="analyze image", context={"image_paths": ["diagram.png"]}))
    assert result.status == AgentStatus.FAILURE
    assert result.errors


def test_vision_preprocessor_preserves_aspect_and_reduces_large_frame(tmp_path) -> None:
    from PIL import Image, ImageDraw
    from tools.vision import VisionPreprocessor

    source = tmp_path / "pid.png"; output = tmp_path / "optimized.png"
    image = Image.new("RGB", (3200, 1800), "white")
    ImageDraw.Draw(image).rectangle((400, 300, 2800, 1500), outline="black", width=8)
    image.save(source)
    prepared = VisionPreprocessor(target_long_side=1200, max_long_side=1400).process(source, output)
    assert prepared.processed_width <= 1400
    assert prepared.processed_height <= 1400
    assert abs((prepared.processed_width / prepared.processed_height) - (2400 / 1200)) < 0.05
    assert prepared.margin_removed is True
    assert prepared.metadata()["original_width"] == 3200


@pytest.mark.asyncio
async def test_vision_agent_uses_one_preprocessed_call(tmp_path) -> None:
    from unittest.mock import AsyncMock, MagicMock
    from PIL import Image

    image = tmp_path / "diagram.png"; Image.new("RGB", (120, 80), "white").save(image)
    provider = MagicMock(); provider.encode_images.return_value = ["encoded"]
    provider.chat = AsyncMock(return_value=MagicMock(content="pump P-101 and valve V-1"))
    agent = OllamaSpecialistAgent(
        AgentDescriptor(name="vision_agent", role="vision", capabilities=[AgentCapability.VISION], provider_name="stub"),
        provider,
    )
    result = await agent.run(AgentRequest(task="analyze P&ID", context={"image_paths": [str(image)]}))
    assert result.status == AgentStatus.SUCCESS
    assert result.metadata["vision_calls"] == 1
    provider.chat.assert_awaited_once()


def test_success_criteria_string_coerced_to_single_element_list() -> None:
    """A single string for success_criteria must NOT be split character-by-character."""
    req = AgentRequest(task="run tests", success_criteria="all unit tests pass")
    assert req.success_criteria == ["all unit tests pass"]
    assert req.success_criteria != list("all unit tests pass")

    # When already a list of strings, preserve the elements
    req_list = AgentRequest(task="run tests", success_criteria=["criterion 1", "criterion 2"])
    assert req_list.success_criteria == ["criterion 1", "criterion 2"]

    # When empty or whitespace
    req_empty = AgentRequest(task="run tests", success_criteria="   ")
    assert req_empty.success_criteria == []


@pytest.mark.asyncio
async def test_master_preserves_single_string_success_criteria_in_delegation() -> None:
    """When planner outputs a string for success_criteria, Master wraps it as one list item."""
    registry = AgentRegistry()
    agent = StubAgent("coder", [AgentCapability.CODING], AgentResult(status=AgentStatus.SUCCESS))
    registry.register(agent)

    async def planner(request: str, capabilities: dict[str, list[str]]) -> list[dict]:
        return [{
            "agent": "coder",
            "task": "run tests",
            "success_criteria": "all tests must pass with 0 failures",
            "constraints": "read-only workspace",
        }]

    state = await MasterAgent(registry, planner=planner).run("test task")
    assert len(agent.requests) == 1
    delegated_req = agent.requests[0]
    # Ensure it's not split into ['a', 'l', 'l', ' ', 't', ...]
    assert delegated_req.success_criteria == ["all tests must pass with 0 failures"]
    assert delegated_req.constraints == ["read-only workspace"]


def test_agent_result_rejects_literal_template_and_generates_id() -> None:
    """AgentResult never retains literal '{{agent_execution_id}}'."""
    # 1. Default creation generates execution ID
    res = AgentResult(status=AgentStatus.SUCCESS)
    assert res.agent_execution_id.startswith("exec_")
    assert res.agent_execution_id != "{{agent_execution_id}}"

    # 2. When model output JSON has literal placeholder "{{agent_execution_id}}", validator replaces it
    model_output = {
        "agent_execution_id": "{{agent_execution_id}}",
        "status": "success",
        "summary": "Completed",
    }
    validated = AgentResult.model_validate(model_output)
    assert validated.agent_execution_id.startswith("exec_")
    assert validated.agent_execution_id != "{{agent_execution_id}}"


@pytest.mark.asyncio
async def test_specialist_agent_enforces_infrastructure_execution_id() -> None:
    """Specialist agent overwrites model-invented execution ID with infrastructure ID."""
    from unittest.mock import AsyncMock, MagicMock
    from runtime.agents import OllamaSpecialistAgent

    descriptor = AgentDescriptor(
        name="test_agent", role="tester", capabilities=[AgentCapability.CODING],
        provider_name="mock", allowed_tools=[],
    )
    provider = MagicMock()
    # Model attempts to return a literal template placeholder or invented ID
    provider.generate = AsyncMock(return_value=MagicMock(
        content='{"agent_execution_id": "{{agent_execution_id}}", "status": "success", "summary": "done"}'
    ))
    agent = OllamaSpecialistAgent(descriptor, provider)

    req = AgentRequest(task="inspect", agent_execution_id="exec_infrastructure_42")
    result = await agent.run(req)

    # Result must have the infrastructure-assigned execution ID, not model output
    assert result.agent_execution_id == "exec_infrastructure_42"
    assert result.agent_execution_id != "{{agent_execution_id}}"


@pytest.mark.asyncio
async def test_master_propagates_generated_execution_id_to_subtasks_and_results() -> None:
    """MasterAgent generates execution IDs, propagates to AgentResult, and logs in trace."""
    registry = AgentRegistry()
    agent = StubAgent("coder", [AgentCapability.CODING], AgentResult(status=AgentStatus.SUCCESS, summary="ok"))
    registry.register(agent)

    trace: list[dict] = []
    async def planner(request: str, capabilities: dict[str, list[str]]) -> list[dict]:
        return [{"agent": "coder", "task": "step 1", "success_criteria": "success"}]

    state = await MasterAgent(registry, planner=planner, trace=trace).run("run task")

    assert len(state.subtasks) == 1
    assert len(state.agent_results) == 1

    subtask_id = state.subtasks[0].agent_execution_id
    result_id = state.agent_results[0].agent_execution_id

    assert subtask_id.startswith("exec_")
    assert subtask_id != "{{agent_execution_id}}"
    # Propagated to AgentResult
    assert subtask_id == result_id

    # Verify trace events recorded the generated agent_execution_id
    events_with_exec_id = [
        e for e in trace
        if e.get("event") in {"MASTER_DELEGATE", "AGENT_START", "AGENT_COMPLETE", "MASTER_REVIEW"}
    ]
    assert len(events_with_exec_id) >= 4
    for e in events_with_exec_id:
        assert e.get("agent_execution_id") == result_id

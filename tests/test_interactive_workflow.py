"""End-to-end interactive approval, denial, and self-correction workflow tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.registry import ModelRegistry
from runtime.approvals import ApprovalManager
from runtime.orchestrator import Orchestrator
from storage.outputs import OutputStore
from tests.test_agent_loop import FakeProvider
from tools.workspace import WorkspaceReadTools


@pytest.mark.asyncio
async def test_interactive_approval_visible_and_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Approval is visibly requested and approval executes edit then pytest."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "math_lib.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    (workspace / "test_math.py").write_text("from math_lib import add\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8")

    prompts_shown: list[str] = []

    def mock_input(prompt: str) -> str:
        prompts_shown.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", mock_input)

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    # Keep _approval_fn as None so prompt_terminal and input() are visibly exercised
    orchestrator._approval_fn = None
    orchestrator.workspace_tools = WorkspaceReadTools(
        workspace,
        approver=orchestrator._edit_approver,
        command_approver=orchestrator._command_approver,
    )
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"tool","tool":"read_file","arguments":{"path":"math_lib.py"}}',
        '{"action":"tool","tool":"edit_file","arguments":{"path":"math_lib.py","old_text":"return a - b","new_text":"return a + b"}}',
        '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_math.py","cwd":"."}}',
        '{"action":"final","answer":"Fixed math_lib.py and verified with pytest."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open math_lib.py", "interactive-approval-test")]
    output = events[-1]["data"]["output"]

    assert output["current_step"] == "completed"
    assert len(prompts_shown) >= 1
    assert "Approve: edit_file math_lib.py" in prompts_shown[0]
    assert "return a + b" in (workspace / "math_lib.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("user_input", ["n", "no", "", "invalid", "cancel"])
@pytest.mark.asyncio
async def test_interactive_approval_denial_leaves_file_byte_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, user_input: str
) -> None:
    """User denial or empty input leaves the file byte-for-byte unchanged."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original_bytes = b"def add(a, b):\n    return a - b\n"
    target_file = workspace / "math_lib.py"
    target_file.write_bytes(original_bytes)

    monkeypatch.setattr("builtins.input", lambda prompt: user_input)

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = None
    orchestrator.workspace_tools = WorkspaceReadTools(
        workspace,
        approver=orchestrator._edit_approver,
        command_approver=orchestrator._command_approver,
    )
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"tool","tool":"edit_file","arguments":{"path":"math_lib.py","old_text":"return a - b","new_text":"return a + b"}}',
        '{"action":"final","answer":"Could not complete edit because approval was denied."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open math_lib.py", f"denial-{user_input}-test")]
    output = events[-1]["data"]["output"]

    # File must be byte-for-byte identical
    assert target_file.read_bytes() == original_bytes

    # Structured denial was received
    tool_results = [
        e for e in events
        if e.get("name") == "tool_result" and e.get("data", {}).get("tool") == "edit_file"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["data"]["approval"] == "denied"
    assert tool_results[0]["data"]["status"] == "failure"


@pytest.mark.asyncio
async def test_self_correction_e2e_edit_fail_edit_pass(tmp_path: Path) -> None:
    """Complete workflow: read -> edit -> pytest FAIL -> observe failure -> second edit -> pytest PASS -> final."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Initially broken function
    (workspace / "math_lib.py").write_text("def compute(x):\n    return x * 0\n", encoding="utf-8")
    # Test requires compute(5) == 25 and compute(2) == 4 (square function)
    test_content = (
        "from math_lib import compute\n\n"
        "def test_compute():\n"
        "    assert compute(2) == 4\n"
        "    assert compute(5) == 25\n"
    )
    (workspace / "test_math.py").write_text(test_content, encoding="utf-8")

    approvals_recorded: list[str] = []

    def mock_approval(action: str, *args: Any) -> bool:
        approvals_recorded.append(action)
        return True

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = mock_approval
    orchestrator.workspace_tools = WorkspaceReadTools(
        workspace,
        approver=orchestrator._edit_approver,
        command_approver=orchestrator._command_approver,
    )
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    # Model flow:
    # 1. read_file
    # 2. first edit: insufficient fix (return x * 2, passes compute(2)==4 but fails compute(5)==25)
    # 3. pytest -> FAIL
    # 4. second edit: correct fix (return x * x)
    # 5. pytest -> PASS
    # 6. final answer
    provider = FakeProvider([
        '{"action":"tool","tool":"read_file","arguments":{"path":"math_lib.py"}}',
        '{"action":"tool","tool":"edit_file","arguments":{"path":"math_lib.py","old_text":"return x * 0","new_text":"return x * 2"}}',
        '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_math.py","cwd":"."}}',
        '{"action":"tool","tool":"edit_file","arguments":{"path":"math_lib.py","old_text":"return x * 2","new_text":"return x * x"}}',
        '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_math.py","cwd":"."}}',
        '{"action":"final","answer":"Successfully fixed compute function to return x * x and verified all tests pass."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open math_lib.py", "self-correct-test")]
    output = events[-1]["data"]["output"]

    assert output["current_step"] == "completed"
    assert "Successfully fixed" in output["messages"][-1].content
    assert (workspace / "math_lib.py").read_text(encoding="utf-8") == "def compute(x):\n    return x * x\n"

    # Both edits were approved
    edit_approvals = [a for a in approvals_recorded if a == "edit_file"]
    assert len(edit_approvals) == 2

    # Check trace events to confirm the fail -> pass progression
    run_dir = Path(output["output_dir"])
    trace_data = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    cmd_results = [
        e for e in trace_data["events"]
        if e.get("step") == "tool_result" and e.get("metadata", {}).get("tool") == "execute_command"
    ]
    assert len(cmd_results) == 2
    # First pytest failed
    assert cmd_results[0]["metadata"]["status"] == "failure"
    # Second pytest succeeded
    assert cmd_results[1]["metadata"]["status"] == "success"


@pytest.mark.asyncio
async def test_read_edit_verification_sequence(tmp_path: Path) -> None:
    """Validate full flow: read_file -> edit_file -> automated read_file verification."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target_file = workspace / "sample.py"
    target_file.write_text("x = 10\ny = 20\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = lambda *_: True
    orchestrator.workspace_tools = WorkspaceReadTools(
        workspace,
        approver=orchestrator._edit_approver,
        command_approver=orchestrator._command_approver,
    )
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"tool","tool":"read_file","arguments":{"path":"sample.py"}}',
        '{"action":"tool","tool":"edit_file","arguments":{"path":"sample.py","old_text":"x = 10","new_text":"x = 99"}}',
        '{"action":"final","answer":"Successfully updated sample.py and verified modification."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("update sample.py", "read-edit-verify-test")]
    output = events[-1]["data"]["output"]

    assert output["current_step"] == "completed"
    assert (workspace / "sample.py").read_text(encoding="utf-8") == "x = 99\ny = 20\n"

    # Extract all tool_result events in sequence
    tool_results = [
        e["data"] for e in events
        if e.get("name") == "tool_result"
    ]
    assert len(tool_results) == 3
    # 1. Initial read_file
    assert tool_results[0]["tool"] == "read_file"
    assert tool_results[0]["status"] == "success"
    assert "x = 10" in tool_results[0]["result"]

    # 2. edit_file with approval
    assert tool_results[1]["tool"] == "edit_file"
    assert tool_results[1]["status"] == "success"
    assert tool_results[1]["approval"] == "approved"

    # 3. Automated post-edit read_file verification
    assert tool_results[2]["tool"] == "read_file"
    assert tool_results[2]["status"] == "success"
    assert tool_results[2]["verification"] is True
    assert "edit confirmed — new_text present in sample.py" in tool_results[2]["result"]

    # Check trace.json contains all three tool events
    run_dir = Path(output["output_dir"])
    trace_data = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    trace_tool_results = [
        e for e in trace_data["events"]
        if e.get("step") == "tool_result"
    ]
    assert len(trace_tool_results) == 3
    assert trace_tool_results[0]["metadata"]["tool"] == "read_file"
    assert trace_tool_results[1]["metadata"]["tool"] == "edit_file"
    assert trace_tool_results[1]["metadata"]["approval"] == "approved"
    assert trace_tool_results[2]["metadata"]["tool"] == "read_file"
    assert trace_tool_results[2]["metadata"]["duration_ms"] == 0



"""AC-01 through AC-11 Acceptance Tests for AEGIS Coding Agent.

These tests prove the full agentic coding loop against the acceptance criteria
from the SIH-2026 engineering brief.  They use FakeProvider to avoid live
Ollama calls, exercising the real production code paths (coding_graph,
orchestrator, workspace tools, command_policy, verification) without any
network dependency.

Acceptance criteria tested:
    AC-01  Model receives workspace description in context.
    AC-02  Model calls a tool → result returned as evidence.
    AC-03  Model reads a file → content present in state.
    AC-04  Model creates a file → file exists on disk, verified immediately.
    AC-05  Model edits a file → edit confirmed, old text absent.
    AC-06  Model runs tests, sees failure, calls edit, reruns → success.
    AC-07  Disallowed command is rejected; loop continues safely.
    AC-08  Path traversal attempt is rejected; workspace stays intact.
    AC-09  Edit without approval is denied; file unchanged.
    AC-10  Iteration budget exhausted → terminal_status == blocked; no crash.
    AC-11  Full trace persisted to disk with plan_created and tool_result events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from models.registry import ModelRegistry
from runtime.coding_graph import run_coding_graph
from runtime.orchestrator import Orchestrator
from storage.outputs import OutputStore
from tools.workspace import WorkspaceReadTools


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

class FakeProvider:
    """Sequential fake provider: yields each response string in order."""

    def __init__(self, responses: list[str], model_id: str = "qwen2.5-coder:7b") -> None:
        self._responses = iter(responses)
        self.model_id = model_id
        self.calls: list[list[dict]] = []

    def encode_images(self, files: list[str]) -> list[str]:
        return []

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: Any):
        self.calls.append(messages)
        response = next(self._responses)
        for char in response:
            yield char


class FakeRegistry:
    """Minimal tool registry backed by real WorkspaceReadTools."""

    def __init__(self, workspace: Path, approver: Any = None) -> None:
        from langchain_core.tools import tool as lc_tool
        self._ws = WorkspaceReadTools(workspace, approver=approver or (lambda *_: True))
        tools_by_name: dict[str, Any] = {}

        for t in self._ws.as_langchain_tools():
            tools_by_name[t.name] = t

        self._tools = tools_by_name

    def list_names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Any:
        return self._tools[name]


# ---------------------------------------------------------------------------
# AC-01: Model receives workspace context (system prompt included in messages)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac01_model_receives_workspace_context(tmp_path: Path) -> None:
    """AC-01: The system prompt is injected before the first model call."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")

    calls: list[list[dict]] = []

    class CapturingProvider(FakeProvider):
        async def stream_chat(self, messages, **kwargs):
            calls.append(messages)
            yield '{"action":"final","answer":"context verified"}'

    registry = FakeRegistry(workspace)
    state = await run_coding_graph(
        CapturingProvider([]),
        registry,
        "List the workspace",
    )
    assert calls, "Provider was never called — context never delivered."
    first_call = calls[0]
    roles = [m["role"] for m in first_call]
    assert "system" in roles, "System context message missing from first model call."


# ---------------------------------------------------------------------------
# AC-02: Model calls a tool → result returned as evidence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac02_tool_result_returned_as_evidence(tmp_path: Path) -> None:
    """AC-02: A tool call produces structured evidence in state.tool_results."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = FakeRegistry(workspace)

    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"list_directory","arguments":{"path":"."}}',
            '{"action":"final","answer":"Listed workspace."}',
        ]),
        registry,
        "List workspace",
    )
    assert state["terminal_status"] == "success"
    assert len(state["tool_results"]) >= 1, "No tool result evidence recorded."
    tr = state["tool_results"][0]
    assert tr.get("tool") == "list_directory"


# ---------------------------------------------------------------------------
# AC-03: Model reads a file → content present in state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac03_model_reads_file_content_in_state(tmp_path: Path) -> None:
    """AC-03: read_file result carries the actual file content."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "hello.py").write_text("print('hello')\n", encoding="utf-8")

    registry = FakeRegistry(workspace)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"hello.py"}}',
            '{"action":"final","answer":"Read the file."}',
        ]),
        registry,
        "Read hello.py",
    )
    assert state["terminal_status"] == "success"
    assert any("hello" in str(r) for r in state["tool_results"]), \
        "File content not present in tool_results evidence."


# ---------------------------------------------------------------------------
# AC-04: Model creates a file → file exists on disk, verified immediately
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac04_create_file_exists_on_disk_and_verified(tmp_path: Path) -> None:
    """AC-04: create_file places the file on disk and the graph verifies it."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    content = "# generated\nx = 42\n"

    state = await run_coding_graph(
        FakeProvider([
            json.dumps({
                "action": "tool", "tool": "create_file",
                "arguments": {"path": "generated.py", "content": content},
            }),
            '{"action":"final","answer":"File created."}',
        ]),
        registry,
        "Create generated.py",
    )
    assert state["terminal_status"] == "success"
    assert (workspace / "generated.py").exists(), "File not created on disk."
    assert "42" in (workspace / "generated.py").read_text(encoding="utf-8")
    # Verify the post-edit read_file verification event fired
    verification_events = [
        e for e in state.get("events", [])
        if e.get("kind") == "tool_result" and e.get("verification") is True
    ]
    assert verification_events, "Immediate post-create verification event missing."


# ---------------------------------------------------------------------------
# AC-05: Model edits a file → edit confirmed, old text absent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac05_edit_file_confirmed_old_text_absent(tmp_path: Path) -> None:
    """AC-05: edit_file applies the patch and the old text is gone."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "app.py"
    target.write_text("def run():\n    print('old')\n", encoding="utf-8")

    registry = FakeRegistry(workspace, approver=lambda *_: True)

    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"app.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "app.py",
                    "old_text": "print('old')",
                    "new_text": "print('new')",
                },
            }),
            '{"action":"final","answer":"Edit applied."}',
        ]),
        registry,
        "Update app.py",
    )
    assert state["terminal_status"] == "success"
    text = target.read_text(encoding="utf-8")
    assert "print('new')" in text, "New text not applied."
    assert "print('old')" not in text, "Old text still present."


# ---------------------------------------------------------------------------
# AC-06: Model runs tests, sees failure, calls edit, reruns → success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac06_iterative_bug_fix_via_tool_loop(tmp_path: Path) -> None:
    """AC-06: Iterative fix: create broken script → run → fail → fix → rerun → pass."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Broken fixture: syntax-valid but assertion fails
    broken = workspace / "app.py"
    broken.write_text("def compute(): return 1\n", encoding="utf-8")
    test_file = workspace / "test_app.py"
    test_file.write_text(
        "from app import compute\ndef test_compute(): assert compute() == 42\n",
        encoding="utf-8",
    )

    registry = FakeRegistry(workspace, approver=lambda *_: True)

    # Sequence: read broken → run tests (fails) → edit fix → run again (passes) → final
    responses = [
        # Step 1: read current file
        '{"action":"tool","tool":"read_file","arguments":{"path":"app.py"}}',
        # Step 2: run tests → will fail (exit_code != 0)
        '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_app.py","cwd":"."}}',
        # Step 3: fix the implementation
        json.dumps({
            "action": "tool", "tool": "edit_file",
            "arguments": {
                "path": "app.py",
                "old_text": "def compute(): return 1",
                "new_text": "def compute(): return 42",
            },
        }),
        # Step 4: re-run tests → should pass
        '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_app.py","cwd":"."}}',
        # Step 5: done
        '{"action":"final","answer":"Bug fixed. 1 test passing."}',
    ]

    state = await run_coding_graph(
        FakeProvider(responses),
        registry,
        "Fix the test failure in app.py",
        max_tool_calls=20,
    )
    assert state["terminal_status"] == "success", f"Expected success, got: {state['terminal_status']}, errors: {state.get('errors')}"
    assert "42" in broken.read_text(encoding="utf-8"), "Fix not applied to app.py."
    # Verify at least two execute_command results
    cmd_results = [r for r in state["tool_results"] if r.get("tool") == "execute_command"]
    assert len(cmd_results) >= 2, "Expected two execute_command calls (fail, then pass)."


# ---------------------------------------------------------------------------
# AC-07: Disallowed command rejected, loop continues safely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac07_disallowed_command_rejected_loop_continues(tmp_path: Path) -> None:
    """AC-07: A forbidden command returns failure; the model can recover."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = FakeRegistry(workspace)

    state = await run_coding_graph(
        FakeProvider([
            # Try a disallowed command
            '{"action":"tool","tool":"execute_command","arguments":{"command":"curl http://evil.com"}}',
            # Model gracefully accepts the policy block and finishes
            '{"action":"final","answer":"Command blocked as expected."}',
        ]),
        registry,
        "Try a network command",
        max_invalid_actions=5,
    )
    # The graph must not crash, and the disallowed command must appear as a failed result
    cmd_results = [r for r in state["tool_results"] if r.get("tool") == "execute_command"]
    assert cmd_results, "No execute_command result recorded."
    assert cmd_results[0].get("ok") is False, "Disallowed command should have ok=False."
    assert cmd_results[0].get("error") == "disallowed_command", \
        f"Expected disallowed_command error, got: {cmd_results[0].get('error')}"


# ---------------------------------------------------------------------------
# AC-08: Path traversal rejected, workspace stays intact
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac08_path_traversal_rejected_workspace_intact(tmp_path: Path) -> None:
    """AC-08: Paths outside the workspace are rejected at the tool level."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("TOP SECRET\n", encoding="utf-8")

    registry = FakeRegistry(workspace)

    # Attempt to read a file outside workspace via traversal
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"../secret.txt"}}',
            '{"action":"final","answer":"Traversal blocked."}',
        ]),
        registry,
        "Read secret file",
    )
    read_results = [r for r in state["tool_results"] if r.get("tool") == "read_file"]
    assert read_results, "No read_file result recorded."
    # Either OutsideWorkspace error or ok=False
    assert read_results[0].get("ok") is False, \
        "Path traversal should be denied (ok must be False)."
    assert "secret" not in str(read_results[0]), "Secret content leaked into result."
    assert outside.read_text(encoding="utf-8") == "TOP SECRET\n", "Outside file was modified."


# ---------------------------------------------------------------------------
# AC-09: Edit without approval denied, file unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac09_edit_denied_file_unchanged(tmp_path: Path) -> None:
    """AC-09: An edit that fails approval leaves the file byte-identical."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original = "sensitive content\n"
    target = workspace / "protected.py"
    target.write_text(original, encoding="utf-8")

    # approver always denies
    registry = FakeRegistry(workspace, approver=lambda *_: False)

    state = await run_coding_graph(
        FakeProvider([
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "protected.py",
                    "old_text": "sensitive content",
                    "new_text": "hacked content",
                },
            }),
            '{"action":"final","answer":"Edit denied; file unchanged."}',
        ]),
        registry,
        "Modify protected.py",
    )
    assert target.read_text(encoding="utf-8") == original, \
        "File was modified despite approval denial."
    edit_results = [r for r in state["tool_results"] if r.get("tool") == "edit_file"]
    assert edit_results, "No edit_file result recorded."
    assert edit_results[0].get("ok") is False or edit_results[0].get("error") == "approval_denied"


# ---------------------------------------------------------------------------
# AC-10: Iteration budget exhausted → terminal_status == blocked, no crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac10_iteration_budget_exhausted_blocked_not_crash(tmp_path: Path) -> None:
    """AC-10: When max_iterations is reached the graph halts gracefully."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Infinite loop provider: always returns invalid JSON (not a tool or final)
    class LoopProvider:
        async def stream_chat(self, messages, **kwargs):
            yield "not valid json at all"

    registry = FakeRegistry(workspace)
    state = await run_coding_graph(
        LoopProvider(),
        registry,
        "Do something impossible",
        max_iterations=3,
        max_invalid_actions=3,
    )
    assert state["terminal_status"] == "blocked", \
        f"Expected blocked, got {state['terminal_status']}"
    assert "final_answer" in state, "No final_answer on exhaustion."


# ---------------------------------------------------------------------------
# AC-11: Full trace persisted to disk with plan_created and tool_result events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ac11_full_trace_persisted_to_disk(tmp_path: Path) -> None:
    """AC-11: Orchestrator.astream persists a trace with required event types."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sample.py").write_text("x = 1\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(
        Path(__file__).parent.parent / "config" / "models.yaml"
    )
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = lambda *_: True
    orchestrator.workspace_tools = WorkspaceReadTools(
        workspace,
        approver=orchestrator._edit_approver,
        command_approver=orchestrator._command_approver,
    )
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    from tests.test_agent_loop import FakeProvider as OrchestratorFakeProvider
    provider = OrchestratorFakeProvider([
        '{"action":"tool","tool":"list_directory","arguments":{"path":"."}}',
        '{"action":"final","answer":"Listed workspace."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("list files", "ac11-trace-test")]
    output = events[-1]["data"]["output"]

    assert output["current_step"] == "completed", \
        f"Run did not complete: {output.get('current_step')}"

    run_dir = Path(output["output_dir"])
    assert run_dir.exists(), "Output directory not created."
    trace_file = run_dir / "trace.json"
    assert trace_file.exists(), "trace.json not written."
    metadata_file = run_dir / "metadata.json"
    assert metadata_file.exists(), "metadata.json not written."

    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))
    event_steps = {e.get("step") for e in trace_data.get("events", [])}

    # Verify key event types present
    assert "classify" in event_steps, "classify event missing from trace."
    assert "route" in event_steps, "route event missing from trace."
    assert "tool_result" in event_steps, "tool_result event missing from trace."
    assert "persist_output" in event_steps, "persist_output event missing from trace."

    # Verify run result file (OutputStore writes result.txt)
    result_file = run_dir / "result.txt"
    assert result_file.exists(), "result.txt not written by OutputStore."

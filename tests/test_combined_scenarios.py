"""Combined tool scenario tests for AEGIS.

These tests exercise multi-tool sequences — the realistic patterns that
emerge in real use: read → edit → verify, command failure → diagnose → fix,
path-traversal-then-recover, approval-denied-then-retry, etc.

All tests use FakeProvider / real WorkspaceReadTools / real command_policy,
so no Ollama required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from runtime.coding_graph import run_coding_graph
from tools.workspace import WorkspaceReadTools


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FakeProvider:
    """Sequential fake provider."""

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
    """Tool registry backed by real WorkspaceReadTools."""

    def __init__(self, workspace: Path, approver: Any = None) -> None:
        self._ws = WorkspaceReadTools(workspace, approver=approver or (lambda *_: True))
        self._tools = {t.name: t for t in self._ws.as_langchain_tools()}

    def list_names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Any:
        return self._tools[name]


# ---------------------------------------------------------------------------
# Scenario 1: Read → Edit → Verify (post-edit read-back)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_read_edit_verify_readback(tmp_path: Path) -> None:
    """Full read → edit → re-read-back sequence; old text absent, new present."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.py").write_text("DEBUG = False\nVERSION = '1.0'\n", encoding="utf-8")

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"config.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "config.py",
                    "old_text": "DEBUG = False",
                    "new_text": "DEBUG = True",
                },
            }),
            '{"action":"tool","tool":"read_file","arguments":{"path":"config.py"}}',
            '{"action":"final","answer":"DEBUG flag enabled."}',
        ]),
        registry,
        "Enable debug mode in config.py",
    )
    assert state["terminal_status"] == "success"
    text = (workspace / "config.py").read_text(encoding="utf-8")
    assert "DEBUG = True" in text
    assert "DEBUG = False" not in text
    # Post-edit verification event
    ver_events = [e for e in state["events"] if e.get("kind") == "tool_result" and e.get("verification")]
    assert ver_events, "No post-edit verification event recorded."


# ---------------------------------------------------------------------------
# Scenario 2: List → Find → Read (discovery chain)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_list_find_read_chain(tmp_path: Path) -> None:
    """list_directory → find_files → read_file discovery sequence."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    src = workspace / "src"
    src.mkdir()
    (src / "utils.py").write_text("def helper(): return 42\n", encoding="utf-8")
    (workspace / "main.py").write_text("from src.utils import helper\n", encoding="utf-8")

    registry = FakeRegistry(workspace)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"list_directory","arguments":{"path":"."}}',
            '{"action":"tool","tool":"find_files","arguments":{"pattern":"*.py"}}',
            '{"action":"tool","tool":"read_file","arguments":{"path":"src/utils.py"}}',
            '{"action":"final","answer":"Found helper in src/utils.py."}',
        ]),
        registry,
        "Find the helper function",
    )
    assert state["terminal_status"] == "success"
    read_results = [r for r in state["tool_results"] if r.get("tool") == "read_file"]
    assert read_results, "read_file result missing."
    assert "helper" in str(read_results[0])


# ---------------------------------------------------------------------------
# Scenario 3: Command success → edit → command rerun (full fix cycle)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_command_fail_edit_rerun(tmp_path: Path) -> None:
    """Run command (fail) → edit → rerun (pass): canonical iterative fix cycle."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "math.py").write_text("def square(n): return n * n * n  # BUG\n", encoding="utf-8")
    (workspace / "test_math.py").write_text(
        "from math import square\ndef test_square(): assert square(3) == 9\n",
        encoding="utf-8",
    )

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"math.py"}}',
            '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_math.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "math.py",
                    "old_text": "def square(n): return n * n * n  # BUG",
                    "new_text": "def square(n): return n * n",
                },
            }),
            '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_math.py"}}',
            '{"action":"final","answer":"Bug fixed. Tests pass."}',
        ]),
        registry,
        "Fix the bug in math.py",
        max_tool_calls=20,
    )
    assert state["terminal_status"] == "success"
    assert "n * n * n" not in (workspace / "math.py").read_text(encoding="utf-8")
    cmd_results = [r for r in state["tool_results"] if r.get("tool") == "execute_command"]
    assert len(cmd_results) >= 2, "Expected at least two command invocations."


# ---------------------------------------------------------------------------
# Scenario 4: Disallowed → retry with allowed → success
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_disallowed_then_allowed_command(tmp_path: Path) -> None:
    """Disallowed command → policy block → model retries with allowed command."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("x = 1\n", encoding="utf-8")

    registry = FakeRegistry(workspace)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"execute_command","arguments":{"command":"rm app.py"}}',
            '{"action":"tool","tool":"read_file","arguments":{"path":"app.py"}}',
            '{"action":"final","answer":"Read app.py after blocked deletion."}',
        ]),
        registry,
        "Remove app.py",
        max_invalid_actions=5,
    )
    # rm is blocked
    blocked = [r for r in state["tool_results"] if r.get("tool") == "execute_command" and not r.get("ok")]
    assert blocked, "rm should have been blocked."
    # File still exists
    assert (workspace / "app.py").exists(), "File should not have been deleted."
    # Read succeeded after block
    reads = [r for r in state["tool_results"] if r.get("tool") == "read_file" and r.get("ok")]
    assert reads, "read_file should have succeeded after the blocked rm."


# ---------------------------------------------------------------------------
# Scenario 5: Path traversal → continue safely → read valid file
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_traversal_then_valid_read(tmp_path: Path) -> None:
    """Traversal attempt → blocked → model reads a valid workspace file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.env"
    outside.write_text("API_KEY=super_secret\n", encoding="utf-8")
    (workspace / "readme.txt").write_text("safe content\n", encoding="utf-8")

    registry = FakeRegistry(workspace)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"../secret.env"}}',
            '{"action":"tool","tool":"read_file","arguments":{"path":"readme.txt"}}',
            '{"action":"final","answer":"Read readme.txt."}',
        ]),
        registry,
        "Show me the readme",
    )
    # Traversal blocked
    traversal = [r for r in state["tool_results"] if r.get("tool") == "read_file" and not r.get("ok")]
    assert traversal, "Traversal should have been blocked."
    # Secret never read
    assert "super_secret" not in str(state)
    # Valid file read succeeded
    valid = [r for r in state["tool_results"] if r.get("tool") == "read_file" and r.get("ok")]
    assert valid, "Valid read should have succeeded."
    assert "safe content" in str(valid[0])


# ---------------------------------------------------------------------------
# Scenario 6: Create → immediate re-read (duplicate create blocked)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_create_then_read_no_duplicate(tmp_path: Path) -> None:
    """create_file → read_file: creation succeeded, read returns content."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    content = "# new module\ndef greet(): return 'hello'\n"
    state = await run_coding_graph(
        FakeProvider([
            json.dumps({
                "action": "tool", "tool": "create_file",
                "arguments": {"path": "greet.py", "content": content},
            }),
            '{"action":"tool","tool":"read_file","arguments":{"path":"greet.py"}}',
            '{"action":"final","answer":"greet.py created and verified."}',
        ]),
        registry,
        "Create greet.py",
    )
    assert state["terminal_status"] == "success"
    assert (workspace / "greet.py").exists()
    reads = [r for r in state["tool_results"] if r.get("tool") == "read_file" and r.get("ok")]
    assert reads, "read_file after create should succeed."
    assert "greet" in str(reads[0])


# ---------------------------------------------------------------------------
# Scenario 7: Edit with wrong old_text → failure → read → correct edit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_edit_wrong_old_text_then_correct(tmp_path: Path) -> None:
    """Wrong old_text → exact-match failure → model reads → correct edit."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calc.py").write_text("def add(a, b):\n    return a - b  # bug\n", encoding="utf-8")

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    state = await run_coding_graph(
        FakeProvider([
            # Wrong old_text — won't match
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "calc.py",
                    "old_text": "return a + b  # wrong guess",
                    "new_text": "return a + b",
                },
            }),
            # Model reads the file to find the actual text
            '{"action":"tool","tool":"read_file","arguments":{"path":"calc.py"}}',
            # Correct edit
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "calc.py",
                    "old_text": "return a - b  # bug",
                    "new_text": "return a + b  # fixed",
                },
            }),
            '{"action":"final","answer":"Fixed subtraction bug."}',
        ]),
        registry,
        "Fix the bug in calc.py",
    )
    text = (workspace / "calc.py").read_text(encoding="utf-8")
    assert "a + b" in text
    assert "a - b" not in text
    # First edit failed, second succeeded
    edits = [r for r in state["tool_results"] if r.get("tool") == "edit_file"]
    assert len(edits) >= 2
    failed = [e for e in edits if not e.get("ok")]
    passed = [e for e in edits if e.get("ok")]
    assert failed, "First (wrong old_text) edit should have failed."
    assert passed, "Second (correct) edit should have succeeded."


# ---------------------------------------------------------------------------
# Scenario 8: search_files → read matching file → targeted edit
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_search_read_edit(tmp_path: Path) -> None:
    """search_files finds the bug location, then read → edit."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "network.py").write_text(
        "# networking module\ndef connect(host): return False  # TODO: implement\n",
        encoding="utf-8",
    )
    (workspace / "other.py").write_text("x = 1\n", encoding="utf-8")

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"search_files","arguments":{"query":"TODO"}}',
            '{"action":"tool","tool":"read_file","arguments":{"path":"network.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "network.py",
                    "old_text": "def connect(host): return False  # TODO: implement",
                    "new_text": "def connect(host): return True",
                },
            }),
            '{"action":"final","answer":"Implemented connect in network.py."}',
        ]),
        registry,
        "Implement the TODO in network.py",
    )
    assert state["terminal_status"] == "success"
    text = (workspace / "network.py").read_text(encoding="utf-8")
    assert "return True" in text
    assert "TODO" not in text


# ---------------------------------------------------------------------------
# Scenario 9: Multi-file edit sequence (two files, both verified)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_two_file_edit_both_verified(tmp_path: Path) -> None:
    """Edit two separate files in the same session; both are updated."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.py").write_text("X = 1\n", encoding="utf-8")
    (workspace / "b.py").write_text("Y = 2\n", encoding="utf-8")

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"a.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {"path": "a.py", "old_text": "X = 1", "new_text": "X = 100"},
            }),
            '{"action":"tool","tool":"read_file","arguments":{"path":"b.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {"path": "b.py", "old_text": "Y = 2", "new_text": "Y = 200"},
            }),
            '{"action":"final","answer":"Updated both constants."}',
        ]),
        registry,
        "Update constants in a.py and b.py",
        max_tool_calls=20,
    )
    assert state["terminal_status"] == "success"
    assert "X = 100" in (workspace / "a.py").read_text(encoding="utf-8")
    assert "Y = 200" in (workspace / "b.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Scenario 10: git_status → read → edit → git_status again
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_git_status_edit_git_status(tmp_path: Path) -> None:
    """git_status → edit file → git_status again shows the file modified."""
    import subprocess
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Init a real git repo so git_status works
    subprocess.run(["git", "init", str(workspace)], capture_output=True, check=False)
    subprocess.run(["git", "-C", str(workspace), "config", "user.email", "test@test.com"],
                   capture_output=True, check=False)
    subprocess.run(["git", "-C", str(workspace), "config", "user.name", "Test"],
                   capture_output=True, check=False)
    (workspace / "version.py").write_text("VERSION = '0.1'\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace), "add", "version.py"],
                   capture_output=True, check=False)
    subprocess.run(["git", "-C", str(workspace), "commit", "-m", "init"],
                   capture_output=True, check=False)

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"git_status","arguments":{}}',
            '{"action":"tool","tool":"read_file","arguments":{"path":"version.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "version.py",
                    "old_text": "VERSION = '0.1'",
                    "new_text": "VERSION = '0.2'",
                },
            }),
            '{"action":"tool","tool":"git_status","arguments":{}}',
            '{"action":"final","answer":"Version bumped to 0.2."}',
        ]),
        registry,
        "Bump version to 0.2",
        max_tool_calls=20,
    )
    assert state["terminal_status"] == "success"
    assert "0.2" in (workspace / "version.py").read_text(encoding="utf-8")
    # Both git_status calls present
    git_results = [r for r in state["tool_results"] if r.get("tool") == "git_status"]
    assert len(git_results) >= 2


# ---------------------------------------------------------------------------
# Scenario 11: Approval denied mid-sequence; file unchanged, loop continues
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_approval_denied_midloop_file_unchanged(tmp_path: Path) -> None:
    """Edit denied mid-loop; the protected file stays unchanged; agent can still read."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    protected = workspace / "secrets.py"
    protected.write_text("KEY = 'secret'\n", encoding="utf-8")
    (workspace / "safe.py").write_text("VALUE = 1\n", encoding="utf-8")

    # Deny edits to secrets.py, allow everything else
    def selective_approver(action: str, path: str, *args: Any) -> bool:
        return "secrets" not in path

    registry = FakeRegistry(workspace, approver=selective_approver)
    state = await run_coding_graph(
        FakeProvider([
            # Try to edit the protected file
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "secrets.py",
                    "old_text": "KEY = 'secret'",
                    "new_text": "KEY = 'hacked'",
                },
            }),
            # Fall back to reading a safe file
            '{"action":"tool","tool":"read_file","arguments":{"path":"safe.py"}}',
            '{"action":"final","answer":"Edit denied; read safe.py instead."}',
        ]),
        registry,
        "Modify secrets.py",
        max_invalid_actions=5,
    )
    assert protected.read_text(encoding="utf-8") == "KEY = 'secret'\n", \
        "Protected file should be unchanged."
    denied = [r for r in state["tool_results"] if r.get("tool") == "edit_file" and not r.get("ok")]
    assert denied, "Edit to secrets.py should have been denied."


# ---------------------------------------------------------------------------
# Scenario 12: Budget exhausted on repeated invalid JSON → blocked, not crash
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_repeated_invalid_json_budget_exhausted(tmp_path: Path) -> None:
    """Repeated non-JSON responses hit the invalid_action budget and stop gracefully."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class NoisyProvider:
        model_id = "qwen2.5-coder:7b"
        def encode_images(self, files): return []
        async def stream_chat(self, messages, **kwargs):
            yield "this is not json at all, just prose"

    registry = FakeRegistry(workspace)
    state = await run_coding_graph(
        NoisyProvider(),
        registry,
        "Do something",
        max_iterations=5,
        max_invalid_actions=3,
    )
    assert state["terminal_status"] == "blocked"
    assert "final_answer" in state


# ---------------------------------------------------------------------------
# Scenario 13: py_compile policy — syntax check allowed without approval
# ---------------------------------------------------------------------------

def test_scenario_py_compile_policy_no_approval_required() -> None:
    """python -m py_compile foo.py is allowed by policy without approval."""
    from runtime.command_policy import evaluate_command
    import pathlib
    ws = pathlib.Path("/workspace")
    decision = evaluate_command("python -m py_compile fibonacci.py", ws, ws)
    assert decision.allowed is True
    assert decision.requires_approval is False
    assert decision.command_name == "py_compile"


# ---------------------------------------------------------------------------
# Scenario 14: pytest command allowed without approval
# ---------------------------------------------------------------------------

def test_scenario_pytest_command_no_approval() -> None:
    """pytest and python -m pytest are allowed without approval."""
    from runtime.command_policy import evaluate_command
    import pathlib
    ws = pathlib.Path("/workspace")
    for cmd in ("pytest", "pytest -q tests/", ".venv/bin/pytest", "python -m pytest", ".venv/bin/python -m pytest"):
        d = evaluate_command(cmd, ws, ws)
        assert d.allowed is True, f"pytest command should be allowed: {cmd}"
        assert d.requires_approval is False, f"pytest should not require approval: {cmd}"


# ---------------------------------------------------------------------------
# Scenario 15: Mutating command requires approval (mkdir)
# ---------------------------------------------------------------------------

def test_scenario_mutating_command_requires_approval() -> None:
    """mkdir requires approval per policy."""
    from runtime.command_policy import evaluate_command
    import pathlib
    ws = pathlib.Path("/workspace")
    decision = evaluate_command("mkdir new_dir", ws, ws)
    assert decision.allowed is True
    assert decision.requires_approval is True


# ---------------------------------------------------------------------------
# Scenario 16: Shell injection attempt blocked even inside pytest args
# ---------------------------------------------------------------------------

def test_scenario_shell_injection_in_pytest_args_blocked() -> None:
    """Shell operators inside otherwise-allowed commands are still blocked."""
    from runtime.command_policy import evaluate_command
    import pathlib
    ws = pathlib.Path("/workspace")
    for malicious in (
        "pytest && rm -rf /",
        "pytest; rm file",
        "pytest | cat /etc/passwd",
        "pytest > /etc/cron.d/evil",
    ):
        d = evaluate_command(malicious, ws, ws)
        assert d.allowed is False, f"Should be blocked: {malicious}"


# ---------------------------------------------------------------------------
# Scenario 17: Absolute path outside workspace blocked in command args
# ---------------------------------------------------------------------------

def test_scenario_absolute_path_outside_workspace_blocked() -> None:
    """Absolute path arguments outside workspace are rejected."""
    from runtime.command_policy import evaluate_command
    import pathlib
    ws = pathlib.Path("/workspace")
    decision = evaluate_command("cat /etc/passwd", ws, ws)
    assert decision.allowed is False


# ---------------------------------------------------------------------------
# Scenario 18: Combined — search + edit + create_file + execute_command
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scenario_search_edit_create_run(tmp_path: Path) -> None:
    """Full pipeline: search → edit existing → create new file → run test."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "logic.py").write_text("def add(a, b): return a - b\n", encoding="utf-8")

    registry = FakeRegistry(workspace, approver=lambda *_: True)
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"search_files","arguments":{"query":"def add"}}',
            '{"action":"tool","tool":"read_file","arguments":{"path":"logic.py"}}',
            json.dumps({
                "action": "tool", "tool": "edit_file",
                "arguments": {
                    "path": "logic.py",
                    "old_text": "def add(a, b): return a - b",
                    "new_text": "def add(a, b): return a + b",
                },
            }),
            json.dumps({
                "action": "tool", "tool": "create_file",
                "arguments": {
                    "path": "test_logic.py",
                    "content": "from logic import add\ndef test_add(): assert add(2, 3) == 5\n",
                },
            }),
            '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_logic.py"}}',
            '{"action":"final","answer":"All done: bug fixed, test added, tests pass."}',
        ]),
        registry,
        "Fix add function and add a test for it",
        max_tool_calls=20,
    )
    assert state["terminal_status"] == "success"
    assert "a + b" in (workspace / "logic.py").read_text(encoding="utf-8")
    assert (workspace / "test_logic.py").exists()
    cmd_results = [r for r in state["tool_results"] if r.get("tool") == "execute_command"]
    assert cmd_results, "execute_command result missing."


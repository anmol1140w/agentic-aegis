"""Focused tests for controlled workspace execute_command capability."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage

from models.registry import ModelRegistry
from runtime.command_policy import evaluate_command
from runtime.orchestrator import Orchestrator
from storage.outputs import OutputStore
from tests.test_agent_loop import FakeProvider
from tools.workspace import WorkspaceReadTools


def test_allowed_pytest_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    test_file = workspace / "test_sample.py"
    test_file.write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    tools = WorkspaceReadTools(workspace)
    # Using python -m pytest with relative path
    res = tools.execute_command(".venv/bin/python -m pytest test_sample.py", cwd=".")

    assert res["tool"] == "execute_command"
    assert res["status"] == "success"
    assert res["ok"] is True
    assert res["exit_code"] == 0
    assert "1 passed" in res["stdout"] or "passed" in res["stdout"]


def test_disallowed_command_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tools = WorkspaceReadTools(workspace)

    # Denied binary
    res1 = tools.execute_command("sudo ls")
    assert res1["ok"] is False
    assert res1["status"] == "failure"
    assert res1["error"] == "disallowed_command"
    assert res1["exit_code"] is None

    # Shell chaining attempt
    res2 = tools.execute_command("ls; cat /etc/passwd")
    assert res2["ok"] is False
    assert res2["status"] == "failure"
    assert res2["error"] == "disallowed_command"

    # Redirection attempt
    res3 = tools.execute_command("ls > out.txt")
    assert res3["ok"] is False
    assert res3["status"] == "failure"
    assert res3["error"] == "disallowed_command"

    # Network attempt
    res4 = tools.execute_command("curl http://example.com")
    assert res4["ok"] is False
    assert res4["status"] == "failure"
    assert res4["error"] == "disallowed_command"


def test_mutating_command_requires_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Without approval -> denied
    tools_no_appr = WorkspaceReadTools(workspace, approver=lambda *_: False)
    res1 = tools_no_appr.execute_command("mkdir subfolder")
    assert res1["ok"] is False
    assert res1["status"] == "failure"
    assert res1["error"] == "approval_denied"
    assert not (workspace / "subfolder").exists()

    # With approval -> executes
    tools_appr = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res2 = tools_appr.execute_command("mkdir subfolder")
    assert res2["ok"] is True
    assert res2["status"] == "success"
    assert res2["exit_code"] == 0
    assert (workspace / "subfolder").is_dir()


def test_cwd_outside_workspace_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tools = WorkspaceReadTools(workspace)

    # Relative path escaping root
    res1 = tools.execute_command("pwd", cwd="../")
    assert res1["ok"] is False
    assert res1["status"] == "failure"
    assert res1["error"] == "outside_workspace"

    # Absolute path outside root
    outside = tmp_path / "other"
    outside.mkdir()
    res2 = tools.execute_command("pwd", cwd=str(outside))
    assert res2["ok"] is False
    assert res2["status"] == "failure"
    assert res2["error"] == "outside_workspace"


def test_command_timeout_is_controlled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=5, output="partial test output")

    monkeypatch.setattr(subprocess, "run", fake_run)

    tools = WorkspaceReadTools(workspace)
    res = tools.execute_command("pytest", timeout=5)

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "command_timeout"
    assert "timed out" in res["message"]
    assert res["exit_code"] is None
    assert "partial test output" in res["stdout"]


def test_stdout_stderr_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    large_output = "X" * 30000
    fake_proc = MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = large_output
    fake_proc.stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: fake_proc)

    tools = WorkspaceReadTools(workspace)
    res = tools.execute_command("pytest")

    assert res["ok"] is True
    assert res["truncated"] is True
    assert len(res["stdout"]) <= 12000


def test_nonzero_exit_is_structured(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fail_test = workspace / "test_fail.py"
    fail_test.write_text("def test_broken():\n    assert False\n", encoding="utf-8")

    tools = WorkspaceReadTools(workspace)
    res = tools.execute_command(".venv/bin/python -m pytest test_fail.py", cwd=".")

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["exit_code"] != 0
    assert "1 failed" in res["stdout"] or "failed" in res["stdout"] or "FAILURES" in res["stdout"]


@pytest.mark.asyncio
async def test_command_trace_is_sanitized(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_dummy.py").write_text("def test_ok(): pass\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = lambda *_: True
    orchestrator.workspace_tools = WorkspaceReadTools(workspace, approver=orchestrator._edit_approver, command_approver=orchestrator._command_approver)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"tool","tool":"execute_command","arguments":{"command":"pytest test_dummy.py","cwd":"."}}',
        '{"action":"final","answer":"Tests executed."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open test_dummy.py", "cmd-trace-test")]
    output = events[-1]["data"]["output"]
    run_dir = Path(output["output_dir"])
    trace_data = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))

    cmd_results = [e for e in trace_data["events"] if e.get("step") == "tool_result" and e.get("metadata", {}).get("tool") == "execute_command"]
    assert len(cmd_results) >= 1
    meta = cmd_results[0]["metadata"]
    assert meta["tool"] == "execute_command"
    assert "command" in meta["arguments"]
    assert meta["arguments"]["command"] == "pytest test_dummy.py"


@pytest.mark.asyncio
async def test_execute_command_returns_result_to_agent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "test_a.py").write_text("def test_a(): assert True\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = lambda *_: True
    orchestrator.workspace_tools = WorkspaceReadTools(workspace, approver=orchestrator._edit_approver, command_approver=orchestrator._command_approver)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"tool","tool":"execute_command","arguments":{"command":".venv/bin/python -m pytest test_a.py","cwd":"."}}',
        '{"action":"final","answer":"All 1 test passed in test_a.py."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open test_a.py", "agent-cmd-test")]
    output = events[-1]["data"]["output"]

    assert output["current_step"] == "completed"
    assert "All 1 test passed" in output["messages"][-1].content


@pytest.mark.parametrize(
    "cmd",
    [
        "bash",
        "sh",
        "zsh",
        "python",
        "python -c 'print(1)'",
        "curl http://localhost",
        "wget http://localhost",
        "nc -l 1234",
        "ssh localhost",
        "sudo ls",
        "rm somefile.py",
    ],
)
def test_command_policy_forbidden_binaries(tmp_path: Path, cmd: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = WorkspaceReadTools(workspace)

    res = tools.execute_command(cmd)
    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "disallowed_command"
    assert res["exit_code"] is None


@pytest.mark.parametrize(
    "cmd",
    [
        "pytest && rm file",
        "pytest ; rm file",
        "pytest | other",
        "pytest > file",
        "$(command)",
        "`command`",
    ],
)
def test_command_policy_shell_bypasses(tmp_path: Path, cmd: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tools = WorkspaceReadTools(workspace)

    res = tools.execute_command(cmd)
    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "disallowed_command"
    assert "not permitted" in res["message"]
    assert res["exit_code"] is None


@pytest.mark.parametrize(
    "cmd",
    [
        "pwd",
        "ls",
        "find .",
        "grep -r test .",
        "pytest",
        ".venv/bin/python -m pytest",
        "git status",
        "git diff",
    ],
)
def test_command_policy_allowed_commands(tmp_path: Path, cmd: str) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    decision = evaluate_command(cmd, workspace, workspace)
    assert decision.allowed is True
    assert decision.requires_approval is False


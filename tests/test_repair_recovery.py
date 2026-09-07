"""Regression coverage for bounded, evidence-driven coding recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.agents import AgentCapability, AgentDescriptor, AgentStatus, OllamaSpecialistAgent
from tools.workspace import WorkspaceReadTools


class _Provider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def generate(self, prompt: str):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return type("Response", (), {"content": response})()


def _agent(ws: Path, provider: _Provider) -> OllamaSpecialistAgent:
    tools = WorkspaceReadTools(ws, approver=lambda *_: True)
    descriptor = AgentDescriptor(
        name="coding_agent",
        role="coding specialist",
        capabilities=[AgentCapability.CODING],
        provider_name="local",
        allowed_tools=["read_file", "edit_file", "execute_command", "list_directory", "repository_context"],
    )
    return OllamaSpecialistAgent(
        descriptor,
        provider,
        tools={
            "read_file": tools.read_file,
            "edit_file": tools.edit_file,
            "execute_command": tools.execute_command,
            "list_directory": tools.list_directory,
            "repository_context": tools.repository_context,
        },
    )


def _request(task: str, path: str) -> object:
    return type("Request", (), {
        "task": task,
        "context": {"path": path},
        "constraints": [],
        "success_criteria": [],
        "agent_execution_id": "repair-regression",
    })()


@pytest.mark.asyncio
async def test_repair_uses_new_file_and_verification_evidence(tmp_path: Path) -> None:
    (tmp_path / "fibonacci.py").write_text("func fibonacci(n):\n    return n\n", encoding="utf-8")
    provider = _Provider([
        '{"action":"tool","tool":"execute_command","arguments":{"command":"python -m py_compile fibonacci.py"}}',
        json.dumps({"action": "tool", "tool": "edit_file", "arguments": {
            "path": "fibonacci.py", "old_text": "func fibonacci(n):", "new_text": "def fibonacci(n):"
        }}),
        '{"action":"final","answer":"fixed"}',
    ])

    result = await _agent(tmp_path, provider).run(_request("fix the error in fibonacci.py", "fibonacci.py"))

    assert result.status is AgentStatus.SUCCESS
    state = result.metadata["coding_state"]
    assert "def fibonacci(n):" in state["files_read"]["fibonacci.py"]
    assert state["verification"]["status"] == "passed"
    assert state["commands"][-1]["state_version"] == 1


@pytest.mark.asyncio
async def test_else_if_repair_is_verified_before_completion(tmp_path: Path) -> None:
    (tmp_path / "branch.py").write_text(
        "def choose(flag):\n    if flag:\n        return 1\n    else if not flag:\n        return 0\n",
        encoding="utf-8",
    )
    provider = _Provider([
        '{"action":"tool","tool":"execute_command","arguments":{"command":"python -m py_compile branch.py"}}',
        json.dumps({"action": "tool", "tool": "edit_file", "arguments": {
            "path": "branch.py", "old_text": "else if not flag:", "new_text": "elif not flag:"
        }}),
        '{"action":"final","answer":"fixed"}',
    ])

    result = await _agent(tmp_path, provider).run(_request("fix the error in branch.py", "branch.py"))

    assert result.status is AgentStatus.SUCCESS
    assert result.verification.get("status") == "passed"
    assert "elif not flag:" in (tmp_path / "branch.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_inverse_repair_is_stopped_without_applying_it(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("func broken():\n    else if True:\n        pass\n", encoding="utf-8")
    provider = _Provider([
        '{"action":"tool","tool":"execute_command","arguments":{"command":"python -m py_compile broken.py"}}',
        json.dumps({"action": "tool", "tool": "edit_file", "arguments": {
            "path": "broken.py", "old_text": "func broken():", "new_text": "else if broken():"
        }}),
        json.dumps({"action": "tool", "tool": "edit_file", "arguments": {
            "path": "broken.py", "old_text": "else if broken():", "new_text": "func broken():"
        }}),
    ])

    result = await _agent(tmp_path, provider).run(_request("fix the error in broken.py", "broken.py"))

    assert result.status is AgentStatus.FAILURE
    assert "repair_oscillation" in result.errors
    assert (tmp_path / "broken.py").read_text(encoding="utf-8").startswith("else if broken():")


@pytest.mark.asyncio
async def test_max_tool_steps_is_terminal_failure(tmp_path: Path) -> None:
    provider = _Provider([
        '{"action":"tool","tool":"list_directory","arguments":{"path":"."}}',
    ])

    agent = _agent(tmp_path, provider)
    # Keep this test focused on the orchestration budget rather than on the
    # workspace adapter's path-resolution contract.
    agent.tools["list_directory"] = lambda path=".": {"ok": True, "items": []}
    request = type("Request", (), {
        "task": "inspect the repository",
        "context": {},
        "constraints": [],
        "success_criteria": [],
        "agent_execution_id": "repair-max-steps",
    })()
    result = await agent.run(request)

    assert result.status is AgentStatus.FAILURE
    assert "max tool steps reached" in result.errors
    assert result.verification.get("status") != "passed"

"""Tests for the fix workflow and approval deduplication.

Tests verify:
1. PolicyEngine evaluate handles execute_command policy-defined approval (e.g. py_compile doesn't require approval).
2. Wrapped callables set _APPROVAL_GRANTED so inner approvers don't double-prompt.
3. Diagnostic fix workflow: baseline command failure -> edit -> final automatically verifies
   the fix with py_compile and succeeds without looping or master repair.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from runtime.agents import (
    AgentDescriptor,
    AgentCapability,
    AgentResult,
    AgentStatus,
    OllamaSpecialistAgent,
)
from runtime.tool_policy import PolicyEngine, ToolPolicy, policy_approval_granted
from tools.workspace import WorkspaceReadTools


class SequentialProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.call_count = 0

    async def generate(self, prompt: str) -> Any:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return type("Response", (), {"content": resp})()
        return type("Response", (), {"content": '{"action": "final", "answer": "Done"}'})()


def test_execute_command_policy_in_policy_engine(tmp_path: Path) -> None:
    """PolicyEngine.evaluate delegates to command_policy for execute_command."""
    engine = PolicyEngine(tmp_path)
    engine.register(ToolPolicy(name="execute_command", filesystem="workspace_only", requires_approval=True))

    # py_compile should NOT require approval
    decision = engine.evaluate("execute_command", {"command": "python -m py_compile script.py"})
    assert decision.allowed is True
    assert decision.approval_required is False

    # pytest should NOT require approval
    decision = engine.evaluate("execute_command", {"command": "pytest -q"})
    assert decision.allowed is True
    assert decision.approval_required is False

    # disallowed command (e.g. rm) should be blocked
    decision = engine.evaluate("execute_command", {"command": "rm script.py"})
    assert decision.allowed is False

    # mutating command should require approval
    decision = engine.evaluate("execute_command", {"command": "mkdir new_dir"})
    assert decision.allowed is True
    assert decision.approval_required is True


def test_wrap_callable_sets_approval_granted(tmp_path: Path) -> None:
    """Wrapped callable sets _APPROVAL_GRANTED so inner callback sees it."""
    engine = PolicyEngine(tmp_path)
    prompts = 0

    def mock_requester(req_id: str, name: str, details: str) -> bool:
        nonlocal prompts
        prompts += 1
        return True

    engine.approval_requester = mock_requester
    engine.register(ToolPolicy(name="mutating_tool", requires_approval=True))

    observed_inside = None

    def sample_func(arg: str) -> str:
        nonlocal observed_inside
        observed_inside = policy_approval_granted()
        return f"result: {arg}"

    wrapped = engine.wrap_callables({"mutating_tool": sample_func})["mutating_tool"]
    res = wrapped("test")

    assert res == "result: test"
    assert prompts == 1
    assert observed_inside is True
    # After invocation, the context var is reset
    assert policy_approval_granted() is False


@pytest.mark.asyncio
async def test_diagnostic_fix_workflow_auto_verifies_on_final(tmp_path: Path) -> None:
    """A coding agent fixing syntax errors has its final action verified by py_compile automatically."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    py_file = ws / "calc.py"
    # Broken syntax: func instead of def
    py_file.write_text("func calc(a, b):\n    return a + b\n", encoding="utf-8")

    ws_tools = WorkspaceReadTools(ws, approver=lambda *_: True)

    provider = SequentialProvider([
        # 1. First model action: run py_compile
        '{"action": "tool", "tool": "execute_command", "arguments": {"command": "python -m py_compile calc.py"}}',
        # 2. Second model action: edit the file to fix syntax
        json.dumps({
            "action": "tool",
            "tool": "edit_file",
            "arguments": {
                "path": "calc.py",
                "old_text": "func calc(a, b):",
                "new_text": "def calc(a, b):",
            },
        }),
        # 3. Third model action: final action! (Model didn't rerun py_compile itself)
        '{"action": "final", "answer": "Fixed syntax error in calc.py"}',
    ])

    descriptor = AgentDescriptor(
        name="coding_agent",
        role="coding specialist",
        capabilities=[AgentCapability.CODING],
        provider_name="local",
        allowed_tools=["read_file", "edit_file", "execute_command", "repository_context"],
    )

    agent = OllamaSpecialistAgent(
        descriptor,
        provider,
        tools={
            "read_file": ws_tools.read_file,
            "edit_file": ws_tools.edit_file,
            "execute_command": ws_tools.execute_command,
            "repository_context": ws_tools.repository_context,
        },
    )

    request = type("Req", (), {
        "task": "fix the error in calc.py",
        "context": {"path": "calc.py", "workspace_root": str(ws)},
        "constraints": [],
        "success_criteria": [],
        "agent_execution_id": "test_exec_01",
    })()

    result = await agent.run(request)

    # Result must be SUCCESS, not FAILURE with command_verification_required
    assert result.status == AgentStatus.SUCCESS
    assert result.verification.get("status") == "passed"
    assert "calc.py" in result.changes
    assert "def calc(a, b):" in (ws / "calc.py").read_text(encoding="utf-8")

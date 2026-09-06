"""Focused tests for exclusive CLI slash-command dispatch."""

from __future__ import annotations

import pytest

import cli


class FakeRegistry:
    async def check_availability(self):
        return {"local": True}


@pytest.mark.asyncio
async def test_network_command_never_invokes_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "_print_network", lambda network: calls.append("network"))

    result = await cli._handle_slash_command(" /NETWORK   ", FakeRegistry(), object())

    assert result == "handled"
    assert calls == ["network"]


@pytest.mark.asyncio
async def test_command_case_and_whitespace_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_print_network", lambda network: None)
    assert await cli._handle_slash_command(" /NeTwOrK  ", FakeRegistry(), object()) == "handled"


@pytest.mark.asyncio
async def test_models_command_never_invokes_agent() -> None:
    assert await cli._handle_slash_command("/models", FakeRegistry(), object()) == "handled"


@pytest.mark.asyncio
async def test_sandbox_command_runs_health_check(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSandboxTools:
        def __init__(self, _root, **_kwargs):
            pass

        def sandbox_status(self):
            return {"ready": True, "backend": "local-process-group", "workspace": "/tmp/workspace"}

        def execute_command(self, command):
            if command.startswith("python"):
                return {"ok": True, "exit_code": 0, "stdout": "4\n"}
            return {"ok": False, "error": "disallowed_command"}

    monkeypatch.setattr(cli, "WorkspaceReadTools", FakeSandboxTools)
    assert await cli._handle_slash_command("/sandbox", FakeRegistry(), object()) == "handled"


@pytest.mark.asyncio
async def test_quit_command_never_invokes_agent() -> None:
    assert await cli._handle_slash_command("/quit", FakeRegistry(), object()) == "exit"


@pytest.mark.asyncio
async def test_master_command_selects_master_mode_without_affecting_legacy() -> None:
    assert await cli._handle_slash_command("  /MASTER  ", FakeRegistry(), object()) == "master"

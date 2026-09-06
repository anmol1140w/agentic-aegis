"""Tests for approval-gated workspace edit_file capability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from models.registry import ModelRegistry
from runtime.orchestrator import Orchestrator
from storage.outputs import OutputStore
from tests.test_agent_loop import FakeProvider
from tools.workspace import WorkspaceReadTools


def test_edit_file_changes_exact_single_match(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "main.py"
    file.write_text("def run():\n    print('old')\n", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("main.py", "print('old')", "print('new')")

    assert res["status"] == "success"
    assert res["changed"] is True
    assert "print('new')" in file.read_text(encoding="utf-8")
    assert "print('old')" not in file.read_text(encoding="utf-8")


def test_edit_file_rejects_missing_old_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "main.py"
    file.write_text("content here", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("main.py", "nonexistent_text", "replacement")

    assert res["status"] == "failure"
    assert res["error"] == "old_text_not_found"
    assert file.read_text(encoding="utf-8") == "content here"


def test_edit_file_rejects_ambiguous_old_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "main.py"
    file.write_text("foo bar foo baz", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("main.py", "foo", "qux")

    assert res["status"] == "failure"
    assert res["error"] == "old_text_ambiguous"
    assert file.read_text(encoding="utf-8") == "foo bar foo baz"


def test_edit_file_rejects_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("outside text", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)

    # Absolute path outside root
    res1 = tools.edit_file(str(outside), "outside", "inside")
    assert res1["status"] == "failure"
    assert res1["error"] == "outside_workspace"

    # Traversal path escaping root
    res2 = tools.edit_file("../outside.py", "outside", "inside")
    assert res2["status"] == "failure"
    assert res2["error"] == "outside_workspace"

    assert outside.read_text(encoding="utf-8") == "outside text"


def test_edit_file_rejects_symlink_escape_if_supported(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "target.py"
    outside.write_text("target text", encoding="utf-8")

    link = workspace / "link_to_outside.py"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported on this platform")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("link_to_outside.py", "target", "modified")

    assert res["status"] == "failure"
    assert res["error"] == "outside_workspace"
    assert outside.read_text(encoding="utf-8") == "target text"


def test_edit_file_requires_explicit_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "test.py"
    file.write_text("unapproved edit", encoding="utf-8")

    tools = WorkspaceReadTools(workspace)  # no approver configured
    res = tools.edit_file("test.py", "unapproved", "approved")

    assert res["status"] == "failure"
    assert res["error"] == "approval_denied"
    assert file.read_text(encoding="utf-8") == "unapproved edit"


def test_edit_file_denial_does_not_modify_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "important.py"
    original_content = "important data\nsecurity critical\n"
    file.write_text(original_content, encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: False)  # explicit deny
    res = tools.edit_file("important.py", "security critical", "hacked")

    assert res["status"] == "failure"
    assert res["error"] == "approval_denied"
    assert file.read_text(encoding="utf-8") == original_content


def test_edit_file_success_returns_bounded_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    file = workspace / "app.py"
    content = "x = 10\ny = 20\n"
    file.write_text(content, encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("app.py", "x = 10", "x = 100")

    assert res["status"] == "success"
    assert res["tool"] == "edit_file"
    assert res["changed"] is True
    assert res["bytes_before"] == len(content.encode("utf-8"))
    assert res["bytes_after"] == len("x = 100\ny = 20\n".encode("utf-8"))
    # Ensure full file content is NOT dumped into the result
    assert "content" not in res


@pytest.mark.asyncio
async def test_edit_file_trace_contains_approval_result_without_file_contents(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_old = "SECRET_PASSWORD_OLD_12345"
    secret_new = "SECRET_PASSWORD_NEW_67890"
    file = workspace / "credentials.py"
    file.write_text(f"API_KEY = '{secret_old}'\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator.workspace_tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        f'{{"action":"tool","tool":"edit_file","arguments":{{"path":"credentials.py","old_text":"{secret_old}","new_text":"{secret_new}"}}}}',
        '{"action":"final","answer":"Successfully updated credentials."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open credentials.py", "edit-trace-test")]
    output = events[-1]["data"]["output"]
    run_dir = Path(output["output_dir"])
    trace_data = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))

    tool_results = [e for e in trace_data["events"] if e.get("step") == "tool_result"]
    assert len(tool_results) >= 1
    tr = tool_results[0]
    meta = tr.get("metadata", {})

    assert meta.get("tool") == "edit_file"
    assert meta.get("approval") == "approved"
    assert meta.get("status") == "success"
    assert "old_text_length" in meta.get("arguments", {})
    assert "new_text_length" in meta.get("arguments", {})
    # Content must NOT leak into trace metadata
    assert secret_old not in json.dumps(meta)
    assert secret_new not in json.dumps(meta)


# ---------------------------------------------------------------------------
# Focused CLI approval + verification tests
# ---------------------------------------------------------------------------


def test_cli_approval_yes_executes_edit(tmp_path: Path) -> None:
    """Approver returning True allows the edit to proceed."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "hello.py"
    f.write_text("print('hello')\n", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("hello.py", "print('hello')", "print('world')")

    assert res["ok"] is True
    assert res["status"] == "success"
    assert "print('world')" in f.read_text(encoding="utf-8")


def test_cli_approval_no_does_not_modify(tmp_path: Path) -> None:
    """Approver returning False prevents any file change."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "hello.py"
    original = "print('hello')\n"
    f.write_text(original, encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: False)
    res = tools.edit_file("hello.py", "print('hello')", "print('world')")

    assert res["ok"] is False
    assert res["error"] == "approval_denied"
    assert f.read_text(encoding="utf-8") == original


def test_cli_approval_empty_denies(tmp_path: Path) -> None:
    """Empty/falsy approver response denies the edit."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "hello.py"
    original = "print('hello')\n"
    f.write_text(original, encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: "")
    res = tools.edit_file("hello.py", "print('hello')", "print('world')")

    assert res["ok"] is False
    assert res["error"] == "approval_denied"
    assert f.read_text(encoding="utf-8") == original


def test_edit_requires_approval(tmp_path: Path) -> None:
    """No approver set → approval_denied."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "hello.py"
    original = "print('hello')\n"
    f.write_text(original, encoding="utf-8")

    tools = WorkspaceReadTools(workspace)  # no approver
    res = tools.edit_file("hello.py", "print('hello')", "print('world')")

    assert res["error"] == "approval_denied"
    assert f.read_text(encoding="utf-8") == original


def test_edit_outside_workspace_denied(tmp_path: Path) -> None:
    """Paths outside workspace root are rejected before approval."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("secret data\n", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file(str(outside), "secret", "leaked")

    assert res["ok"] is False
    assert res["error"] == "outside_workspace"
    assert outside.read_text(encoding="utf-8") == "secret data\n"


@pytest.mark.asyncio
async def test_edit_verification_reads_changed_content(tmp_path: Path) -> None:
    """After successful edit, orchestrator tool loop emits verification event."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "app.py"
    f.write_text("x = 1\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = lambda *_: True
    orchestrator.workspace_tools = WorkspaceReadTools(workspace, approver=orchestrator._edit_approver)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"tool","tool":"edit_file","arguments":{"path":"app.py","old_text":"x = 1","new_text":"x = 42"}}',
        '{"action":"final","answer":"Done."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open app.py", "verify-test")]
    # Find verification event
    verify_events = [
        e for e in events
        if e.get("name") == "tool_result"
        and e.get("data", {}).get("verification") is True
    ]
    assert len(verify_events) >= 1
    assert "confirmed" in verify_events[0]["data"]["result"]
    assert f.read_text(encoding="utf-8") == "x = 42\n"


@pytest.mark.asyncio
async def test_edit_verification_failure_is_reported(tmp_path: Path) -> None:
    """If verification fails (e.g. read_file fails or content missing), failure is reported."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "vanish.py"
    f.write_text("old_content\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = lambda *_: True
    orchestrator.workspace_tools = WorkspaceReadTools(workspace, approver=orchestrator._edit_approver)
    # Mock read_file on workspace_tools to simulate verification failure
    orchestrator.workspace_tools.read_file = lambda path: {"ok": False, "error": "ReadError"}
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"tool","tool":"edit_file","arguments":{"path":"vanish.py","old_text":"old_content","new_text":"new_content"}}',
        '{"action":"final","answer":"Done."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open vanish.py", "verify-fail-test")]
    verify_events = [
        e for e in events
        if e.get("name") == "tool_result"
        and e.get("data", {}).get("verification") is True
    ]
    assert len(verify_events) >= 1
    assert verify_events[0]["data"]["status"] == "failure"
    assert "failed to read" in verify_events[0]["data"]["result"]


@pytest.mark.asyncio
async def test_mutation_trace_excludes_file_contents(tmp_path: Path) -> None:
    """Trace metadata must contain only lengths, not actual old/new text."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = "SUPER_SECRET_KEY_XYZ"
    f = workspace / "config.py"
    f.write_text(f"key = '{secret}'\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator._approval_fn = lambda *_: True
    orchestrator.workspace_tools = WorkspaceReadTools(workspace, approver=orchestrator._edit_approver)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        f'{{"action":"tool","tool":"edit_file","arguments":{{"path":"config.py","old_text":"{secret}","new_text":"REDACTED"}}}}',
        '{"action":"final","answer":"Redacted."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open config.py", "trace-sanitize-test")]
    output = events[-1]["data"]["output"]
    run_dir = Path(output["output_dir"])
    trace_json = (run_dir / "trace.json").read_text(encoding="utf-8")

    assert secret not in trace_json
    assert "old_text_length" in trace_json
    assert "new_text_length" in trace_json


def test_edit_file_rejects_empty_old_text(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "test.py"
    f.write_text("content", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("test.py", "", "replacement")

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "empty_old_text"
    assert f.read_text(encoding="utf-8") == "content"


def test_edit_file_rejects_identical_replacement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "test.py"
    f.write_text("hello world", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("test.py", "hello", "hello")

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "identical_replacement"
    assert f.read_text(encoding="utf-8") == "hello world"


def test_edit_file_rejects_large_replacement(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "test.py"
    f.write_text("placeholder", encoding="utf-8")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    huge_text = "A" * (70 * 1024)
    res = tools.edit_file("test.py", "placeholder", huge_text)

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "file_too_large"
    assert f.read_text(encoding="utf-8") == "placeholder"


def test_edit_file_rejects_binary_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "data.bin"
    f.write_bytes(b"\x00\x01\x02\x03hello\x00")

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("data.bin", "hello", "world")

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "binary_file"


def test_edit_file_rejects_missing_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("nonexistent.py", "old", "new")

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "file_not_found"


def test_edit_file_handles_permission_denied(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    f = workspace / "protected.py"
    f.write_text("protected text", encoding="utf-8")

    def fake_read_bytes(*args, **kwargs):
        raise PermissionError("Permission denied: protected.py")

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    tools = WorkspaceReadTools(workspace, approver=lambda *_: True)
    res = tools.edit_file("protected.py", "protected", "modified")

    assert res["ok"] is False
    assert res["status"] == "failure"
    assert res["error"] == "permission_denied"


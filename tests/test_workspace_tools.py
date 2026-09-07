"""Focused security tests for the read-only workspace-agent slice."""

from __future__ import annotations

from pathlib import Path

from tools.workspace import WorkspaceReadTools


def test_workspace_tools_reject_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"; root.mkdir()
    assert WorkspaceReadTools(root).read_file("../../etc/passwd")["error"] == "OutsideWorkspace"


def test_workspace_outside_root_rejected(tmp_path: Path) -> None:
    root = tmp_path / "workspace"; root.mkdir()
    assert WorkspaceReadTools(root).get_file_info(str(tmp_path / "outside.txt"))["error"] == "OutsideWorkspace"


def test_workspace_creation_is_confined_to_root(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    tools = WorkspaceReadTools(root, approver=lambda *_: True)

    rejected = tools.create_file(str(outside), "must stay inside")
    assert rejected["error"] == "OutsideWorkspace"
    assert not outside.exists()

    created = tools.create_file("inside.txt", "inside")
    assert created["ok"] is True
    assert (root / "inside.txt").read_text() == "inside"


def test_workspace_read_tools_are_non_mutating(tmp_path: Path) -> None:
    root = tmp_path / "workspace"; root.mkdir()
    source = root / "notes.txt"; source.write_text("pump guard")
    tools = WorkspaceReadTools(root)

    assert tools.read_file("notes.txt")["content"] == "pump guard"
    assert tools.search_files("guard")["matches"][0]["line"] == 1
    assert tools.find_files("*.txt")["items"][0]["name"] == "notes.txt"
    assert source.read_text() == "pump guard"


def test_read_tool_calls_are_traced(tmp_path: Path) -> None:
    root = tmp_path / "workspace"; root.mkdir()
    (root / "notes.txt").write_text("ok")
    result = WorkspaceReadTools(root).list_directory()
    assert result["ok"] is True
    assert result["tool"] == "list_directory"

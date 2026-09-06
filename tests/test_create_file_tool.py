import pytest
from tools.workspace import WorkspaceReadTools


def test_create_file_is_approval_gated_and_reusable(tmp_path):
    tools = WorkspaceReadTools(tmp_path, approver=lambda *args: True)
    result = tools.create_file("race_stats.py", "print('ok')")
    assert result["ok"] is True
    assert (tmp_path / "race_stats.py").read_text() == "print('ok')"


def test_create_file_denied_and_escape_rejected(tmp_path):
    denied = WorkspaceReadTools(tmp_path, approver=lambda *args: False)
    assert denied.create_file("x.py", "pass")["error"] == "approval_denied"
    allowed = WorkspaceReadTools(tmp_path, approver=lambda *args: True)
    assert allowed.create_file("../../outside.py", "pass")["error"] == "OutsideWorkspace"


def test_create_python_script_is_approved_and_workspace_bound(tmp_path):
    tools = WorkspaceReadTools(tmp_path, approver=lambda *_: True)
    result = tools.create_python_script("race_stats.py", "print('ok')")
    assert result["ok"] is True
    assert result["tool"] == "create_python_script"
    assert (tmp_path / "race_stats.py").read_text() == "print('ok')"
    assert tools.create_python_script("race_stats.txt", "print('ok')")["error"] == "invalid_python_path"
    assert tools.create_python_script("broken.py", "def broken(:\n pass") ["error"] == "syntax_error"

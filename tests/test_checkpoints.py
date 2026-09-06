from pathlib import Path

from storage.checkpoints import WorkspaceCheckpointStore
from tools.workspace import WorkspaceReadTools


def test_checkpoint_diff_and_restore(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    source.mkdir()
    file = source / "note.txt"
    file.write_text("before\n", encoding="utf-8")
    store = WorkspaceCheckpointStore(source)
    checkpoint = store.create("before edit")
    file.write_text("after\n", encoding="utf-8")
    (source / "new.txt").write_text("new\n", encoding="utf-8")
    diff = store.diff(checkpoint["checkpoint_id"])
    assert diff["modified"] == ["note.txt"]
    assert diff["added"] == ["new.txt"]
    assert "-before" in diff["patches"]["note.txt"]
    store.restore(checkpoint["checkpoint_id"])
    assert file.read_text() == "before\n"
    assert not (source / "new.txt").exists()


def test_workspace_checkpoint_is_approval_gated(tmp_path: Path) -> None:
    tools = WorkspaceReadTools(tmp_path, approver=lambda *_: False)
    assert tools.create_checkpoint()["error"] == "approval_denied"
    tools = WorkspaceReadTools(tmp_path, approver=lambda *_: True)
    assert tools.create_checkpoint("approved")["ok"] is True

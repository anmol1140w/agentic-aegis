from pathlib import Path

from routing.classifier import ExecutionMode, TaskClassifier, TaskType
from runtime.actions import parse_action
from tools.workspace import WorkspaceReadTools


def test_tree_returns_recursive_structure_and_skips_generated_dirs(tmp_path: Path) -> None:
    (tmp_path / "src" / "nested").mkdir(parents=True)
    (tmp_path / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "src" / "nested" / "data.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret-ish metadata", encoding="utf-8")
    result = WorkspaceReadTools(tmp_path).tree()
    assert result["ok"] is True
    assert "src/" in result["tree"]
    assert "main.py" in result["tree"]
    assert "data.json" in result["tree"]
    assert ".git" not in result["tree"]


def test_tree_limits_depth_and_entries(tmp_path: Path) -> None:
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "deep.txt").write_text("x", encoding="utf-8")
    result = WorkspaceReadTools(tmp_path).tree(max_depth=1, max_entries=2)
    assert result["entry_count"] <= 2
    assert result["truncated"] is False
    assert any(item.get("truncated") for item in result["entries"] if item["type"] == "directory")


def test_tree_action_and_classifier() -> None:
    assert parse_action('{"action":"tree","path":"src"}') == {
        "action": "tool", "tool": "tree", "arguments": {"path": "src"}
    }
    task = TaskClassifier().classify("show the recursive file structure of src")
    assert task.execution_mode is ExecutionMode.MODEL_WITH_TOOLS
    assert task.task_type is TaskType.CODING

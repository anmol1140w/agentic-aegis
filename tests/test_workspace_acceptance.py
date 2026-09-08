"""Acceptance tests for read-only workspace requests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.registry import ModelRegistry
from routing.classifier import TaskType
from runtime.orchestrator import Orchestrator
from storage.outputs import OutputStore
from tests.test_agent_loop import FakeProvider
from tools.workspace import WorkspaceReadTools


def test_repository_context_is_bounded_and_rooted(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("repository contract\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (root / ".venv").mkdir()
    (root / ".venv" / "secret.txt").write_text("must not enter context\n", encoding="utf-8")

    result = WorkspaceReadTools(root).repository_context()

    assert result["ok"] is True
    assert result["root"] == str(root.resolve())
    assert "README.md" in result["important_files"]
    assert "src/main.py" in result["files"]
    assert not any("secret.txt" in path for path in result["files"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_request,expected_tool,provider_script",
    [
        (
            "list the current directory",
            "list_directory",
            [
                '{"action":"list_directory"}',
                '{"action":"final","answer":"Directory contents listed."}',
            ],
        ),
        (
            "find all Python files containing NetworkMonitor",
            "search_files",
            [
                '{"action":"search_files","query":"NetworkMonitor"}',
                '{"action":"final","answer":"Found NetworkMonitor in security/network.py."}',
            ],
        ),
        (
            "search for document_analysis and explain where it is handled",
            "search_files",
            [
                '{"action":"search_files","query":"document_analysis"}',
                '{"action":"final","answer":"Handled in routing/classifier.py."}',
            ],
        ),
        (
            "read the verify_runs.py file",
            "read_file",
            [
                '{"action":"read_file","path":"verify_runs.py"}',
                '{"action":"final","answer":"verify_runs.py contains test verification utilities."}',
            ],
        ),
        (
            "show me the git status",
            "git_status",
            [
                '{"action":"git_status"}',
                '{"action":"final","answer":"Git status clean."}',
            ],
        ),
    ],
)
async def test_read_only_workspace_acceptance_flow(
    tmp_path: Path,
    user_request: str,
    expected_tool: str,
    provider_script: list[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "verify_runs.py").write_text("# verify runs\n", encoding="utf-8")

    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator.workspace_tools = WorkspaceReadTools(workspace)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider(provider_script)
    provider.config = registry.get_provider("qwen-coder").config
    provider.model_id = registry.get_provider("qwen-coder").model_id
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream(user_request, f"accept-{expected_tool}")]
    output = events[-1]["data"]["output"]

    assert output["current_step"] == "completed"

    run_dir = Path(output["output_dir"])
    trace_data = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))

    # Verify classification
    classify_events = [e for e in trace_data["events"] if e.get("step") == "classify"]
    assert len(classify_events) == 1

    # Verify model
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model"] == registry.get_provider("qwen-coder").model_id
    assert metadata["task_type"] == TaskType.CODING.value

    # Verify tool execution
    tool_results = [e for e in trace_data["events"] if e.get("step") == "tool_result"]
    assert any(e.get("metadata", {}).get("tool") == expected_tool for e in tool_results)

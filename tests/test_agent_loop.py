"""Offline tests for structured model-directed tool loops."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from models.registry import ModelRegistry
from runtime.actions import ActionParseError, parse_action
from runtime.orchestrator import Orchestrator
from routing.classifier import ExecutionMode, TaskClassifier
from storage.outputs import OutputStore
from tools.workspace import WorkspaceReadTools


class FakeProvider:
    def __init__(self, responses: list[str], model_id: str = "qwen2.5-coder:7b", config: Any = None) -> None:
        self.responses = iter(responses)
        self.calls: list[list[dict[str, str]]] = []
        self.model_id = model_id
        self.config = config

    def encode_images(self, files: list[str]) -> list[str]:
        return []

    async def stream_chat(self, messages: list[dict[str, str]], **kwargs: object):
        self.calls.append(messages)
        response = next(self.responses)
        for token in response:
            yield token


@pytest.mark.asyncio
async def test_agent_run_persists_full_trace_and_network_report(tmp_path: Path) -> None:
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(tmp_path / "workspace"))

    events = [event async for event in orchestrator.astream("Calculate 2 + 2", "trace-test")]
    output = events[-1]["data"]["output"]
    run_dir = Path(output["output_dir"])
    trace = json.loads((run_dir / "trace.json").read_text())
    network = json.loads((run_dir / "network_report.json").read_text())

    assert {entry["step"] for entry in trace["events"]} >= {
        "classify", "route", "tool", "network_status", "persist_output"
    }
    assert [entry["step"] for entry in trace["events"]].count("classify") == 1
    assert [entry["step"] for entry in trace["events"]].count("route") == 1
    assert isinstance(network["external_connection_count"], int)
    assert "observed_external_connection_count" in network
    assert network["denied_attempts"] == []


def test_parse_structured_tool_and_final_actions() -> None:
    assert parse_action("```json\n{\"action\":\"tool\",\"tool\":\"list_directory\",\"arguments\":{\"path\":\".\"}}\n```") == {
        "action": "tool", "tool": "list_directory", "arguments": {"path": "."}
    }
    assert parse_action('{"action":"final","answer":"done"}') == {
        "action": "final", "answer": "done"
    }
    assert parse_action('<tool_call>{"name":"inspect_path","arguments":"{\\"path\\":\\".\\"}"}</tool_call>') == {
        "action": "tool", "tool": "inspect_path", "arguments": {"path": "."}
    }
    assert parse_action('<think>planning {"action":"final","answer":"done"}') == {
        "action": "final", "answer": "done"
    }
    assert parse_action('{"tool":"inspect_path","path":"."}') == {
        "action": "tool", "tool": "inspect_path", "arguments": {"path": "."}
    }
    assert parse_action('{"tool_calls":[{"function":{"name":"inspect_path","arguments":"{\\"path\\":\\".\\"}"}}]}') == {
        "action": "tool", "tool": "inspect_path", "arguments": {"path": "."}
    }
    with pytest.raises(ActionParseError):
        parse_action("run list_directory now")


@pytest.mark.asyncio
async def test_tool_loop_uses_multiple_tools_and_actual_results(tmp_path: Path) -> None:
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    notes = workspace / "notes.txt"
    notes.write_text("pump-101 pressure 4 bar", encoding="utf-8")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    provider = FakeProvider([
        '{"action":"tool","tool":"list_directory","arguments":{"path":"' + str(workspace) + '"}}',
        '{"action":"tool","tool":"read_file","arguments":{"path":"' + str(notes) + '"}}',
        '{"action":"final","answer":"The notes report pump-101 at 4 bar."}',
    ])

    events = [event async for event in orchestrator._stream_tool_loop(
        provider, [HumanMessage(content="Inspect the folder and read the notes.")]
    )]

    assert [event["kind"] for event in events] == ["tool", "result", "tool", "result", "final"]
    assert "pump-101 pressure 4 bar" in events[3]["result"]
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_tool_loop_accepts_plain_final_prose_after_tool(tmp_path: Path) -> None:
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(tmp_path / "workspace"))
    provider = FakeProvider([
        '{"tool":"list_directory","path":"' + str(tmp_path) + '"}',
        "The directory was inspected successfully.",
    ])
    events = [event async for event in orchestrator._stream_tool_loop(
        provider, [HumanMessage(content="Inspect the folder.")]
    )]
    assert events[-1] == {
        "kind": "final",
        "answer": "The directory was inspected successfully.",
        "iteration": 2,
        "duration_ms": events[-1]["duration_ms"],
    }


@pytest.mark.asyncio
async def test_workspace_agent_can_complete_search_then_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"; workspace.mkdir()
    (workspace / "router.py").write_text("document_analysis = True")
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    agent = Orchestrator(registry, output_store=OutputStore(workspace))
    agent.workspace_tools = WorkspaceReadTools(workspace)
    agent.tool_registry = agent._build_tool_registry()
    provider = FakeProvider([
        '{"action":"tool","tool":"search_files","arguments":{"query":"document_analysis"}}',
        '{"action":"tool","tool":"read_file","arguments":{"path":"router.py"}}',
        '{"action":"final","answer":"document_analysis is in router.py."}',
    ])

    events = [event async for event in agent._stream_tool_loop(provider, [HumanMessage(content="Find it")])]

    assert [event["kind"] for event in events] == ["tool", "result", "tool", "result", "final"]
    assert "document_analysis" in events[3]["result"]


@pytest.mark.asyncio
async def test_workspace_agent_stops_at_max_steps(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"; workspace.mkdir()
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    agent = Orchestrator(registry, output_store=OutputStore(workspace), max_iterations=99)
    agent.workspace_tools = WorkspaceReadTools(workspace)
    agent.tool_registry = agent._build_tool_registry()
    provider = FakeProvider(['{"action":"tool","tool":"list_directory","arguments":{}}'] * 20)

    events = [event async for event in agent._stream_tool_loop(provider, [HumanMessage(content="List")])]

    assert events[-1]["kind"] == "error"
    assert "Maximum tool-loop iterations (20)" in events[-1]["message"]


@pytest.mark.asyncio
async def test_tool_loop_stops_after_repeated_invalid_actions(tmp_path: Path) -> None:
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(tmp_path / "workspace"), max_iterations=10)
    provider = FakeProvider(["not json", "still not json", "also not json"])

    events = [event async for event in orchestrator._stream_tool_loop(
        provider, [HumanMessage(content="Inspect files.")]
    )]

    assert events[-1]["kind"] == "error"
    assert "invalid tool actions" in events[-1]["message"]
    assert len(provider.calls) == 3


def test_public_filesystem_tools_are_registered(tmp_path: Path) -> None:
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(tmp_path / "workspace"))
    names = set(orchestrator.tool_registry.list_names())
    assert {"list_directory", "find_files", "read_file", "search_files", "get_file_info", "git_status", "git_diff", "extract_ocr"} <= names


def test_edit_file_requires_one_match_and_write_permission(tmp_path: Path) -> None:
    from tools.filesystem import FilesystemTools
    from tools.permissions import AccessMode, SessionPermissions

    target = tmp_path / "notes.txt"
    target.write_text("before")
    prompts: list[AccessMode] = []
    tools = FilesystemTools(
        SessionPermissions(),
        lambda mode, *_: prompts.append(mode) or True,
    )
    result = tools.edit_file(str(target), "before", "after")
    assert result["ok"] is True
    assert target.read_text() == "after"
    assert prompts == [AccessMode.WRITE]


def test_read_file_does_not_decode_binary_documents(tmp_path: Path) -> None:
    from tools.filesystem import FilesystemTools
    from tools.permissions import SessionPermissions

    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.7\nnot text context")
    result = FilesystemTools(SessionPermissions([tmp_path])).read_file(str(document))
    assert result["ok"] is False
    assert result["error"] == "UnsupportedFileType"


def test_explicit_pdf_ocr_is_routed_to_deterministic_tool(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    task = TaskClassifier().classify(f"{pdf} use ocr tool and explain")
    assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
    assert "ocr_pdf" in task.requires_tools


def test_pdf_explanation_is_routed_to_deterministic_unsupported_read(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    task = TaskClassifier().classify(f"{pdf} explain")
    assert task.execution_mode == ExecutionMode.DIRECT_TOOL
    assert task.direct_tool_name == "safe_read_file"


def test_pdf_conversion_tool_routing(tmp_path: Path) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    task = TaskClassifier().classify(f"convert {pdf} to images")
    assert task.execution_mode == ExecutionMode.DIRECT_TOOL
    assert task.direct_tool_name == "convert_pdf_to_images"


# ---------------------------------------------------------------------------
# Qwen-Coder shorthand and workspace tool-loop regression tests
# ---------------------------------------------------------------------------

def test_parse_qwen_readonly_action_shorthand() -> None:
    """Qwen shorthand {"action": "<read_only_tool>"} normalizes to {"action": "tool", ...}."""
    # list_directory with empty arguments
    assert parse_action('{"action":"list_directory"}') == {
        "action": "tool", "tool": "list_directory", "arguments": {}
    }
    # list_directory with explicit arguments
    assert parse_action('{"action":"list_directory","path":"src"}') == {
        "action": "tool", "tool": "list_directory", "arguments": {"path": "src"}
    }
    # git_status
    assert parse_action('{"action":"git_status"}') == {
        "action": "tool", "tool": "git_status", "arguments": {}
    }
    # git_diff
    assert parse_action('{"action":"git_diff"}') == {
        "action": "tool", "tool": "git_diff", "arguments": {}
    }
    # read_file with arguments
    assert parse_action('{"action":"read_file","path":"notes.txt"}') == {
        "action": "tool", "tool": "read_file", "arguments": {"path": "notes.txt"}
    }
    # search_files with arguments
    assert parse_action('{"action":"search_files","query":"def hello"}') == {
        "action": "tool", "tool": "search_files", "arguments": {"query": "def hello"}
    }
    # find_files with arguments
    assert parse_action('{"action":"find_files","pattern":"*.py"}') == {
        "action": "tool", "tool": "find_files", "arguments": {"pattern": "*.py"}
    }
    # get_file_info with arguments
    assert parse_action('{"action":"get_file_info","path":"data.csv"}') == {
        "action": "tool", "tool": "get_file_info", "arguments": {"path": "data.csv"}
    }


def test_parse_unknown_action_is_rejected() -> None:
    """Non-allowlisted actions or mutating actions are rejected."""
    bad_actions = [
        "write_file", "edit_file", "delete_file", "bash", "execute",
        "python_executor", "rm -rf", "drop_table", "arbitrary_cmd", "unknown_action",
    ]
    for bad in bad_actions:
        with pytest.raises(ActionParseError):
            parse_action(f'{{"action":"{bad}"}}')


def test_parse_malformed_action_is_rejected() -> None:
    """Non-object JSON, missing fields, or broken syntax are rejected."""
    # non-object
    with pytest.raises(ActionParseError):
        parse_action("[1, 2, 3]")
    with pytest.raises(ActionParseError):
        parse_action('"string only"')
    with pytest.raises(ActionParseError):
        parse_action("12345")
    # broken syntax
    with pytest.raises(ActionParseError):
        parse_action('{"action": "tool", broken')
    # empty
    with pytest.raises(ActionParseError):
        parse_action("")
    # non-string action
    with pytest.raises(ActionParseError):
        parse_action('{"action": 123}')
    # final without string answer
    with pytest.raises(ActionParseError):
        parse_action('{"action": "final"}')
    with pytest.raises(ActionParseError):
        parse_action('{"action": "final", "answer": 42}')
    # non-dict arguments
    with pytest.raises(ActionParseError):
        parse_action('{"action": "tool", "tool": "list_directory", "arguments": "not_dict"}')


def test_workspace_action_arguments_are_validated() -> None:
    """Tools requiring arguments fail if required arguments are missing."""
    # read_file requires path
    with pytest.raises(ActionParseError):
        parse_action('{"action":"read_file"}')
    with pytest.raises(ActionParseError):
        parse_action('{"action":"tool","tool":"read_file","arguments":{}}')
    # search_files requires query
    with pytest.raises(ActionParseError):
        parse_action('{"action":"search_files"}')
    with pytest.raises(ActionParseError):
        parse_action('{"action":"tool","tool":"search_files","arguments":{}}')
    # find_files requires pattern
    with pytest.raises(ActionParseError):
        parse_action('{"action":"find_files"}')
    with pytest.raises(ActionParseError):
        parse_action('{"action":"tool","tool":"find_files","arguments":{}}')
    # get_file_info requires path
    with pytest.raises(ActionParseError):
        parse_action('{"action":"get_file_info"}')
    with pytest.raises(ActionParseError):
        parse_action('{"action":"tool","tool":"get_file_info","arguments":{}}')
    # Valid arguments succeed
    valid = parse_action('{"action":"read_file","path":"file.py"}')
    assert valid["tool"] == "read_file" and valid["arguments"]["path"] == "file.py"


@pytest.mark.asyncio
async def test_workspace_tool_result_is_returned_to_model(tmp_path: Path) -> None:
    """Tool results are executed and fed back into model message context."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "server.py").write_text("print('hello sovereign')", encoding="utf-8")
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator.workspace_tools = WorkspaceReadTools(workspace)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"list_directory"}',
        '{"action":"final","answer":"Found server.py in workspace."}',
    ])
    events = [event async for event in orchestrator._stream_tool_loop(
        provider, [HumanMessage(content="What files are in the workspace?")]
    )]
    kinds = [e["kind"] for e in events]
    assert kinds == ["tool", "result", "final"]
    assert events[0]["tool"] == "list_directory"
    assert "server.py" in events[1]["result"]
    assert events[2]["answer"] == "Found server.py in workspace."
    # The second model invocation must contain the tool result
    assert len(provider.calls) == 2
    second_call_msgs = provider.calls[1]
    assert any("TOOL RESULT [list_directory]" in m["content"] and "server.py" in m["content"] for m in second_call_msgs)


@pytest.mark.asyncio
async def test_workspace_tool_trace_is_sanitized(tmp_path: Path) -> None:
    """Workspace tool traces contain metadata but do NOT leak full file contents or secrets."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = "TOP_SECRET_PASSWORD_DO_NOT_STORE_IN_TRACE_METADATA_12345"
    (workspace / "credentials.env").write_text(secret, encoding="utf-8")
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    orchestrator.workspace_tools = WorkspaceReadTools(workspace)
    orchestrator.tool_registry = orchestrator._build_tool_registry()

    provider = FakeProvider([
        '{"action":"read_file","path":"credentials.env"}',
        '{"action":"final","answer":"Read the credentials file."}',
    ])
    provider.config = registry.get_provider("qwen-coder").config
    # Inject fake provider into registry for qwen-coder
    registry._providers["qwen-coder"] = provider

    events = [e async for e in orchestrator.astream("open credentials.env", "trace-sanitize-thread")]
    output = events[-1]["data"]["output"]
    run_dir = Path(output["output_dir"])
    trace_file = run_dir / "trace.json"
    assert trace_file.exists()
    trace_data = json.loads(trace_file.read_text(encoding="utf-8"))

    tool_results = [e for e in trace_data["events"] if e.get("step") == "tool_result"]
    assert len(tool_results) >= 1
    tr = tool_results[0]
    meta = tr.get("metadata", {})
    # Check trace schema fields
    assert meta.get("tool") == "read_file"
    assert meta.get("status") == "success"
    assert "duration_ms" in meta
    assert "result_size" in meta and meta["result_size"] > 0
    assert "truncated" in meta
    assert "arguments" in meta
    # Ensure secret is NOT in trace metadata or step message
    assert secret not in json.dumps(meta)
    assert secret not in tr.get("message", "")


@pytest.mark.asyncio
async def test_invalid_tool_action_limit_is_bounded(tmp_path: Path) -> None:
    """Model producing repeated invalid actions halts at 3 attempts."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = ModelRegistry.from_yaml(Path(__file__).parent.parent / "config" / "models.yaml")
    orchestrator = Orchestrator(registry, output_store=OutputStore(workspace))
    provider = FakeProvider([
        '{"action":"invalid_action_1"}',
        '{"action":"invalid_action_2"}',
        '{"action":"invalid_action_3"}',
        '{"action":"final","answer":"never reached"}',
    ])
    events = [e async for e in orchestrator._stream_tool_loop(
        provider, [HumanMessage(content="Do something")]
    )]
    assert events[-1]["kind"] == "error"
    assert "too many invalid tool actions" in events[-1]["message"]
    # Model should have been called exactly 3 times before stopping
    assert len(provider.calls) == 3

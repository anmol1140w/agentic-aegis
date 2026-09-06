"""End-to-end integration tests for the LangGraph orchestrator.

Verifies:
- Direct tool execution (calculator, file ops) without invoking an LLM.
- Phase latency breakdown measurements.
- Serialization safety during LangGraph state transitions.
- Offline execution and error handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.messages import HumanMessage

from models.registry import ModelRegistry
from runtime.orchestrator import Orchestrator
from storage.outputs import OutputStore
from tools import files as file_tools


@pytest.fixture
def test_registry(tmp_path: Path) -> ModelRegistry:
    yaml_content = """
models:
  test-general:
    provider: ollama
    model: "test-general:latest"
    capabilities: [general, reasoning]
    context_length: 8192
    priority: 10

  test-coder:
    provider: ollama
    model: "test-coder:latest"
    capabilities: [coding, debugging]
    context_length: 8192
    priority: 10

  test-vision:
    provider: ollama
    model: "test-vision:latest"
    capabilities: [vision, multimodal, image_analysis]
    context_length: 8192
    priority: 10

  test-fallback:
    provider: ollama
    model: "test-fallback:latest"
    capabilities: [general, lightweight]
    context_length: 4096
    priority: 1
    is_fallback: true
"""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml_content)
    return ModelRegistry.from_yaml(config_path)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    file_tools.set_workspace_root(ws)
    return ws


@pytest.mark.asyncio
async def test_direct_calculator_execution(test_registry: ModelRegistry, tmp_path: Path) -> None:
    """Verify calculator executes directly via tool node in milliseconds without calling LLM."""
    orchestrator = Orchestrator(test_registry, output_store=OutputStore(tmp_path / "workspace"))

    # Prompt: Calculate 234 * 918
    result = await orchestrator.ainvoke("Calculate 234 * 918.", thread_id="test-calc")

    # Verify state and response
    assert result["current_step"] == "completed"
    assert result["is_direct_tool"] is True
    assert result["task"]["direct_tool_name"] == "calculator"

    # Direct tool returns exact numeric answer
    last_msg = result["messages"][-1]
    assert "214812" in str(last_msg.content)

    # Latency should be fast (pure Python tool)
    assert result["tool_ms"] > 0
    assert result["total_ms"] > 0

    # Verify trace entries are dicts (serialization safe)
    assert len(result["trace"]) >= 3
    for entry in result["trace"]:
        assert isinstance(entry, dict)
        assert "step" in entry
        assert "message" in entry

    assert Path(result["output_dir"], "result.txt").read_text().startswith("214812")
    assert Path(result["output_dir"], "metadata.json").exists()
    assert Path(result["output_dir"], "trace.json").exists()


@pytest.mark.asyncio
async def test_direct_file_listing_execution(
    test_registry: ModelRegistry, workspace: Path
) -> None:
    """Verify list_files executes directly via tool node without calling LLM."""
    (workspace / "file_a.txt").write_text("aaa")
    (workspace / "file_b.csv").write_text("bbb")

    orchestrator = Orchestrator(test_registry)
    result = await orchestrator.ainvoke("ls", thread_id="test-files")

    assert result["current_step"] == "completed"
    assert result["is_direct_tool"] is True
    assert result["task"]["direct_tool_name"] == "list_files"

    last_msg = result["messages"][-1]
    content = str(last_msg.content)
    assert "file_a.txt" in content
    assert "file_b.csv" in content


@pytest.mark.asyncio
async def test_direct_read_file_execution(
    test_registry: ModelRegistry, workspace: Path
) -> None:
    """Verify read_file executes directly."""
    (workspace / "notes.txt").write_text("Secret notes content")

    orchestrator = Orchestrator(test_registry)
    result = await orchestrator.ainvoke(
        "read file notes.txt", thread_id="test-read"
    )

    assert result["current_step"] == "completed"
    assert result["is_direct_tool"] is True
    last_msg = result["messages"][-1]
    assert "Secret notes content" in str(last_msg.content)


@pytest.mark.asyncio
async def test_vision_image_path_detection_in_orchestrator(
    test_registry: ModelRegistry, tmp_path: Path
) -> None:
    """Verify image path in prompt is classified as image modality and routed to vision model."""
    img_path = tmp_path / "diagram.png"
    img_path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")

    orchestrator = Orchestrator(test_registry)
    # Just run through classify and route nodes (since Ollama isn't running in unit test)
    state = {
        "messages": [HumanMessage(content=f"{img_path} explain this drawing")],
        "attached_files": [],
        "iteration": 0,
    }
    classify_out = await orchestrator._classify_node(state)
    assert classify_out["task"]["modality"] == "image"
    assert classify_out["task"]["requires_vision"] is True

    route_out = await orchestrator._route_node({**state, **classify_out})
    assert route_out["selected_model"] == "test-vision"
    assert route_out["is_direct_tool"] is False


@pytest.mark.asyncio
async def test_orchestrator_runs_ordered_tool_sequence(test_registry: ModelRegistry, tmp_path: Path) -> None:
    orchestrator = Orchestrator(test_registry, output_store=OutputStore(tmp_path / "workspace"))
    results = await orchestrator.run_tool_sequence([
        {"tool": "calculator", "args": {"expression": "2 + 2"}},
        {"tool": "not_registered", "args": {}},
    ])
    assert results[0] == {"ok": True, "tool": "calculator", "result": "4"}
    assert results[1]["ok"] is False
    assert results[1]["error"] == "KeyError"

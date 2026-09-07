from pathlib import Path

import pytest

from models.base import ModelCapability, ModelConfig
from routing.capability_router import CapabilityTask, choose_model
from runtime.extraction import ExtractedValue, ExtractionReviewQueue
from runtime.tool_policy import PolicyDenied, PolicyEngine, ToolPolicy
from storage.graph_state import SQLiteGraphStateStore
from tools.calculator import calculator
from tools.registry import ToolRegistry


def test_capability_router_enforces_quality_bar() -> None:
    models = [
        ModelConfig(name="vlm", provider="x", model="vlm", capabilities=[ModelCapability.VISION]),
        ModelConfig(name="reasoner", provider="x", model="reasoner", capabilities=[ModelCapability.REASONING]),
    ]
    result = choose_model(CapabilityTask("scan", (ModelCapability.VISION,), 1.0), models)
    assert result.model == "vlm"
    with pytest.raises(LookupError):
        choose_model(CapabilityTask("scan", (ModelCapability.VISION, ModelCapability.REASONING), 1.0), models)


def test_policy_is_deny_by_default_and_bounds_paths(tmp_path: Path) -> None:
    registry = ToolRegistry()
    registry.register(calculator)
    engine = PolicyEngine(tmp_path)
    engine.register(ToolPolicy("calculator", timeout=2, max_output=10))
    assert not engine.evaluate("unknown").allowed
    assert not engine.evaluate("calculator", {"file_path": "/etc/passwd"}).allowed


def test_extraction_review_queue_and_sqlite_snapshot(tmp_path: Path) -> None:
    queue = ExtractionReviewQueue(0.85)
    assert queue.add(ExtractedValue("17.6 bar", 0.7, "report.pdf", page=12))
    assert len(queue.pending()) == 1
    store = SQLiteGraphStateStore(tmp_path / "state.db")
    store.save("run", "task", "checkpoint", {"status": "paused"})
    assert store.load("run")["status"] == "paused"

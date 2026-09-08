from aegis.architecture import ArchitecturePolicy, ExecutionMode
from aegis.evaluation import TrajectoryRecord
from aegis.memory import MemoryKind, MemoryStore


def test_architecture_chooses_least_autonomous_mode():
    policy = ArchitecturePolicy()
    assert policy.decide("calculate 2 + 2").mode == ExecutionMode.DETERMINISTIC
    assert policy.decide("extract text from this PDF").mode == ExecutionMode.WORKFLOW
    decision = policy.decide("edit the repository and run the tests")
    assert decision.mode == ExecutionMode.HYBRID
    assert "tool_allowlist" in decision.required_controls


def test_memory_layers_and_retention(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.record_event("task completed", scope="run-1")
    store.remember("local routing preference", scope="router", kind=MemoryKind.SEMANTIC,
                   provenance="operator", importance=0.9)
    result = store.search("local routing", scope="router")
    assert result[0]["kind"] == "semantic"
    assert result[0]["provenance"] == "operator"


def test_trajectory_scores_tool_behavior():
    score = TrajectoryRecord(
        task_id="t1", expected_tools=["read_file", "execute_command"],
        actual_tools=["read_file", "execute_command"], completed=True,
    ).score()
    assert score["completion"] == 1.0
    assert score["tool_precision"] == 1.0
    assert score["tool_recall"] == 1.0

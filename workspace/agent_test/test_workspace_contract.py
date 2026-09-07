from pathlib import Path


def test_agent_fixture_has_a_local_workspace() -> None:
    workspace = Path(__file__).resolve().parents[1]
    assert workspace.name == "workspace"
    assert (workspace / "fixtures").is_dir()
    assert (workspace / "artifacts").is_dir()
    assert (workspace / "executions").is_dir()

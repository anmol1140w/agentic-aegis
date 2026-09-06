from pathlib import Path

from langchain_core.tools import tool

from storage.artifacts import ArtifactManager
from tools.registry import RiskLevel, ToolRegistry


def test_registry_metadata_discovery_and_permission_filtering():
    @tool
    def read_fixture(path: str) -> str:
        """Read a fixture."""
        return path

    registry = ToolRegistry()
    registry.register(read_fixture, category="filesystem", tags=["files", "read"])
    meta = registry.get_meta("read_fixture")
    assert meta.category == "filesystem"
    assert meta.permissions["read"] is True
    assert registry.search("files")[0].name == "read_fixture"
    assert registry.filter_by_permission("network") == []
    assert registry.check_available("read_fixture") is True


def test_artifact_manager_records_verified_provenance(tmp_path: Path):
    artifact = tmp_path / "report.json"
    artifact.write_text('{"ok": true}', encoding="utf-8")
    manager = ArtifactManager(tmp_path)
    record = manager.register_artifact(artifact, artifact_type="json", execution_id="exec_1", agent_id="document_agent", tool_id="create_document")
    assert record["verification_status"] == "verified"
    assert record["execution_id"] == "exec_1"
    assert manager.find_artifact(record["artifact_id"])["sha256"] == record["sha256"]


def test_artifact_manager_rejects_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("x", encoding="utf-8")
    manager = ArtifactManager(tmp_path / "artifacts")
    try:
        manager.register_artifact(outside)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("artifact escape was accepted")

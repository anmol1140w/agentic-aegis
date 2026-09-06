"""Security tests — path validation, audit logging, network monitoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from security.permissions import PermissionError_, is_path_safe, validate_path
from security.audit import AuditLogger
from security.network import NetworkMonitor, _is_loopback, _is_private


# ---------------------------------------------------------------------------
# Path validation tests
# ---------------------------------------------------------------------------

class TestPathValidation:
    """Test workspace boundary enforcement."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "allowed.txt").write_text("ok")
        return ws

    def test_valid_relative_path(self, workspace: Path) -> None:
        result = validate_path("allowed.txt", workspace, must_exist=True)
        assert result == workspace / "allowed.txt"

    def test_path_traversal_blocked(self, workspace: Path) -> None:
        with pytest.raises(PermissionError_, match="escapes workspace"):
            validate_path("../../etc/passwd", workspace)

    def test_absolute_path_outside_blocked(self, workspace: Path) -> None:
        with pytest.raises(PermissionError_, match="escapes workspace"):
            validate_path("/etc/passwd", workspace)

    def test_blocked_substring(self, workspace: Path) -> None:
        # Create a nested path that contains a blocked segment
        # The resolved path must contain the blocked segment
        with pytest.raises(PermissionError_):
            validate_path("../../root/.ssh/id_rsa", workspace)

    def test_must_exist_fails(self, workspace: Path) -> None:
        with pytest.raises(PermissionError_, match="does not exist"):
            validate_path("nonexistent.txt", workspace, must_exist=True)

    def test_dangerous_extension_blocked_on_write(self, workspace: Path) -> None:
        with pytest.raises(PermissionError_, match="extension"):
            validate_path("script.sh", workspace, allow_write=True)

    def test_safe_extension_allowed_on_write(self, workspace: Path) -> None:
        # .txt should be fine
        result = validate_path("output.txt", workspace, allow_write=True)
        assert result.suffix == ".txt"

    def test_is_path_safe_true(self, workspace: Path) -> None:
        assert is_path_safe("allowed.txt", workspace) is True

    def test_is_path_safe_false(self, workspace: Path) -> None:
        assert is_path_safe("../../etc/passwd", workspace) is False


# ---------------------------------------------------------------------------
# Audit logger tests
# ---------------------------------------------------------------------------

class TestAuditLogger:
    """Test structured audit logging."""

    def test_creates_log_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path)
        audit.log("test_event", task_id="t1")
        audit.close()
        assert log_path.exists()

    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path)
        audit.log("event_a", task_id="t1", model="model-1")
        audit.log("event_b", task_id="t2", status="success")
        audit.close()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "timestamp" in record
            assert "event" in record

    def test_entry_has_all_fields(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        audit = AuditLogger(log_path)
        entry = audit.log(
            "model_invoked",
            task_id="abc",
            model="qwen3:8b",
            tool="calculator",
            status="success",
            duration_ms=42.5,
            metadata={"tokens": 100},
        )
        audit.close()

        assert entry.event == "model_invoked"
        assert entry.task_id == "abc"
        assert entry.model == "qwen3:8b"
        assert entry.duration_ms == 42.5

    def test_context_manager(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit.jsonl"
        with AuditLogger(log_path) as audit:
            audit.log("ctx_event")
        # Should not raise


# ---------------------------------------------------------------------------
# Network monitoring tests
# ---------------------------------------------------------------------------

class TestNetworkMonitor:

    def test_loopback_detection(self) -> None:
        assert _is_loopback("127.0.0.1") is True
        assert _is_loopback("::1") is True
        assert _is_loopback("localhost") is True
        assert _is_loopback("8.8.8.8") is False

    def test_private_detection(self) -> None:
        assert _is_private("192.168.1.1") is True
        assert _is_private("10.0.0.1") is True
        assert _is_private("172.16.0.1") is True
        assert _is_private("8.8.8.8") is False

    def test_snapshot_returns_stats(self) -> None:
        monitor = NetworkMonitor()
        stats = monitor.snapshot()
        assert stats.total_connections >= 0
        assert stats.external_connections >= 0

    def test_snapshot_separates_agent_and_observed_connections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connections = [
            SimpleNamespace(status="ESTABLISHED", pid=101, raddr=SimpleNamespace(ip="127.0.0.1", port=11434)),
            SimpleNamespace(status="ESTABLISHED", pid=101, raddr=SimpleNamespace(ip="8.8.8.8", port=443)),
            SimpleNamespace(status="ESTABLISHED", pid=202, raddr=SimpleNamespace(ip="1.1.1.1", port=443)),
        ]
        monkeypatch.setattr("security.network.psutil.net_connections", lambda kind: connections)

        stats = NetworkMonitor(agent_pid=101).snapshot()

        assert stats.local_connections == 1
        assert stats.external_connections == 1
        assert stats.agent_external_connection_details == [{"remote": "8.8.8.8:443", "status": "ESTABLISHED"}]
        assert stats.observed_external_connections == 1

    def test_counters_increment(self) -> None:
        monitor = NetworkMonitor()
        assert monitor.snapshot().local_model_calls == 0
        monitor.record_model_call()
        monitor.record_model_call()
        monitor.record_tool_call()
        stats = monitor.snapshot()
        assert stats.local_model_calls == 2
        assert stats.local_tool_calls == 1

    def test_format_status(self) -> None:
        monitor = NetworkMonitor()
        status = monitor.format_status()
        assert "Agent network:" in status
        assert "external=" in status
        assert "Observed elsewhere external=" in status

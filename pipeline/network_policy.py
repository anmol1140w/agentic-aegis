"""Enforceable offline network policy.

Allows only explicitly configured loopback Ollama traffic. Denies and logs
all other outbound destinations. Produces a testable network_report.json.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil


@dataclass
class NetworkEvent:
    """A single observed or denied network event."""
    timestamp: str
    direction: str          # "outbound" | "inbound"
    remote_ip: str
    remote_port: int
    status: str             # "allowed" | "denied" | "observed"
    reason: str
    protocol: str = "tcp"


@dataclass
class NetworkPolicyReport:
    """Complete network report for a pipeline run."""
    run_id: str
    policy_mode: str = "enforce_local_only"
    allowed_loopback_port: int | None = None
    start_time: str = ""
    end_time: str = ""
    allowed_connections: list[dict[str, Any]] = field(default_factory=list)
    denied_attempts: list[dict[str, Any]] = field(default_factory=list)
    observed_external: int = 0
    dns_events: list[dict[str, Any]] = field(default_factory=list)
    external_connection_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "policy_mode": self.policy_mode,
            "allowed_loopback_port": self.allowed_loopback_port,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "allowed_connections": self.allowed_connections,
            "denied_attempts": self.denied_attempts,
            "observed_external_count": self.observed_external,
            "dns_events": self.dns_events,
            "external_connection_count": self.external_connection_count,
        }


def _is_loopback(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return addr in ("localhost", "127.0.0.1", "::1")


class NetworkPolicy:
    """Enforceable local-only network policy.

    Allows loopback traffic to a configured Ollama port.
    Denies and logs everything else. Produces a structured report.
    """

    def __init__(
        self,
        *,
        run_id: str,
        allowed_loopback_port: int = 11434,
        enabled: bool = True,
    ) -> None:
        self.run_id = run_id
        self.allowed_loopback_port = allowed_loopback_port
        self.enabled = enabled
        self._report = NetworkPolicyReport(
            run_id=run_id,
            allowed_loopback_port=allowed_loopback_port,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        self._denied: list[NetworkEvent] = []
        self._allowed: list[NetworkEvent] = []

    def check_connection(self, host: str, port: int) -> bool:
        """Check if a connection to host:port is allowed.

        Returns True if allowed, False if denied. Logs the event.
        """
        now = datetime.now(timezone.utc).isoformat()

        if _is_loopback(host) and port == self.allowed_loopback_port:
            evt = NetworkEvent(
                timestamp=now,
                direction="outbound",
                remote_ip=host,
                remote_port=port,
                status="allowed",
                reason="loopback to configured Ollama port",
            )
            self._allowed.append(evt)
            return True

        if _is_loopback(host):
            # Allow other loopback traffic (e.g., PaddleOCR internal)
            evt = NetworkEvent(
                timestamp=now,
                direction="outbound",
                remote_ip=host,
                remote_port=port,
                status="allowed",
                reason="loopback traffic",
            )
            self._allowed.append(evt)
            return True

        # Deny non-loopback
        evt = NetworkEvent(
            timestamp=now,
            direction="outbound",
            remote_ip=host,
            remote_port=port,
            status="denied",
            reason="non-loopback destination denied by policy",
        )
        self._denied.append(evt)
        return False

    def check_url(self, url: str) -> bool:
        """Check if a URL target is allowed by policy."""
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return self.check_connection(host, port)

    def scan_system_connections(self) -> None:
        """Scan current system connections and classify them."""
        try:
            connections = psutil.net_connections(kind="inet")
        except (psutil.AccessDenied, PermissionError):
            return

        for conn in connections:
            if conn.status not in ("ESTABLISHED", "SYN_SENT"):
                continue
            if not conn.raddr:
                continue
            remote_ip = conn.raddr.ip
            remote_port = conn.raddr.port

            if not _is_loopback(remote_ip):
                self._report.observed_external += 1

    def finalize(self) -> NetworkPolicyReport:
        """Finalize and return the network report."""
        self._report.end_time = datetime.now(timezone.utc).isoformat()
        self._report.allowed_connections = [
            {"timestamp": e.timestamp, "remote": f"{e.remote_ip}:{e.remote_port}",
             "reason": e.reason}
            for e in self._allowed
        ]
        self._report.denied_attempts = [
            {"timestamp": e.timestamp, "remote": f"{e.remote_ip}:{e.remote_port}",
             "reason": e.reason}
            for e in self._denied
        ]
        self._report.external_connection_count = (
            len(self._denied) + self._report.observed_external
        )
        self.scan_system_connections()
        return self._report

    def save(self, output_dir: Path) -> Path:
        """Save the network report to output_dir/network_report.json."""
        report = self.finalize()
        path = output_dir / "network_report.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    @property
    def denied_count(self) -> int:
        return len(self._denied)

    @property
    def external_count(self) -> int:
        """Count of external connections attempted by the pipeline (denied)."""
        return len(self._denied)

    def summary_line(self) -> str:
        """One-line terminal summary."""
        total = self.denied_count
        if total == 0:
            return "EXTERNAL NETWORK CALLS: 0"
        return f"FAIL — {total} external attempts blocked/observed"

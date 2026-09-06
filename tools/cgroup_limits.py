"""Best-effort Linux cgroup v2 limits for local command sandboxes.

The limiter is intentionally optional: many developer machines expose cgroup
v2 read-only to unprivileged processes.  In that case the caller can continue
with Bubblewrap's namespace and timeout protections and report that resource
limits were unavailable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path


class CgroupLimitError(RuntimeError):
    """Raised when a cgroup cannot be created or configured."""


class CgroupV2Limit:
    def __init__(
        self,
        *,
        root: str | Path = "/sys/fs/cgroup",
        memory_mb: int = 512,
        max_pids: int = 256,
        cpu_weight: int = 100,
    ) -> None:
        if memory_mb < 1 or max_pids < 1 or not 1 <= cpu_weight <= 10_000:
            raise ValueError("cgroup limits must be positive and cpu_weight must be 1..10000")
        self.root = Path(root)
        self.path = self.root / f"aegis-{uuid.uuid4().hex[:12]}"
        self.memory_mb = memory_mb
        self.max_pids = max_pids
        self.cpu_weight = cpu_weight
        self.created = False

    @property
    def available(self) -> bool:
        return (self.root / "cgroup.controllers").is_file()

    def create(self) -> None:
        if not self.available:
            raise CgroupLimitError("cgroup v2 is not mounted")
        try:
            self.path.mkdir()
            self._write("memory.max", str(self.memory_mb * 1024 * 1024))
            self._write("pids.max", str(self.max_pids))
            self._write("cpu.weight", str(self.cpu_weight))
            self.created = True
        except (OSError, ValueError) as exc:
            self.destroy()
            raise CgroupLimitError(f"cannot configure cgroup limits: {exc}") from exc

    def attach(self, pid: int) -> None:
        if not self.created:
            raise CgroupLimitError("cgroup has not been created")
        self._write("cgroup.procs", str(pid))

    def destroy(self) -> None:
        if not self.path.exists():
            return
        # The kernel removes cgroup pseudo-files with the directory; only the
        # directory itself should be removed after the child has exited.
        # Test doubles and non-kernel cgroup shims expose regular files, so
        # remove those too while leaving real pseudo-files untouched.
        for name in ("memory.max", "pids.max", "cpu.weight", "cgroup.procs"):
            candidate = self.path / name
            try:
                if candidate.is_file() and not candidate.is_fifo():
                    candidate.unlink()
            except OSError:
                pass
        try:
            self.path.rmdir()
        except OSError:
            pass
        self.created = False

    def _write(self, name: str, value: str) -> None:
        (self.path / name).write_text(value, encoding="ascii")


def cgroup_mode() -> str:
    """Return ``auto``, ``on`` or ``off`` from the environment."""
    mode = os.getenv("AEGIS_CGROUP_MODE", "auto").strip().lower()
    return mode if mode in {"auto", "on", "off"} else "auto"

"""Optional native process lifecycle boundary.

The Python layer still owns policy and approvals. When the compiled helper is
available, it owns process-group creation and timeout termination; otherwise
callers use the existing tested Python implementation.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def helper_path() -> Path:
    configured = os.getenv("AEGIS_NATIVE_EXEC")
    if configured:
        return Path(configured).expanduser().resolve()
    name = "aegis-exec.exe" if os.name == "nt" else "aegis-exec"
    return Path(__file__).resolve().parent.parent / "native" / "bin" / name


def available() -> bool:
    path = helper_path()
    return path.is_file() and os.access(path, os.X_OK)


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: int,
        capture_output: bool = True) -> dict[str, Any] | None:
    """Run through the native helper, or return ``None`` when unavailable."""
    if not available():
        return None
    try:
        completed = subprocess.run(
            [str(helper_path()), "--cwd", str(cwd), "--timeout-ms", str(max(1, timeout) * 1000), "--", *command],
            cwd=str(cwd), env=env, capture_output=capture_output, text=True,
            timeout=max(1, timeout) + 5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout or "",
        "stderr": completed.stderr or "",
        "timed_out": completed.returncode == 124,
        "native": True,
    }

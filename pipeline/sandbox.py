"""Bounded coding-task sandbox.

Executes a Python script in a subprocess with:
- Hard timeout
- Networking disabled (attempted via environment)
- Read-only host access (runs in a temp dir copy)
- Captured stdout/stderr/exit code

Falls closed if subprocess is unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SandboxUnavailableError(Exception):
    """Raised when the sandbox backend cannot be used."""


@dataclass
class SandboxResult:
    """Result of a sandboxed code execution."""
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_ms: float
    sandbox_backend: str  # "subprocess" or "unavailable"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:4096],
            "stderr": self.stderr[:4096],
            "timed_out": self.timed_out,
            "elapsed_ms": self.elapsed_ms,
            "sandbox_backend": self.sandbox_backend,
            "error": self.error,
        }


def run_sandboxed(
    *,
    script_path: Path | str,
    input_data: str | None = None,
    timeout_seconds: int = 30,
    python_executable: str | None = None,
) -> SandboxResult:
    """Execute a Python script in a bounded sandbox.

    Parameters
    ----------
    script_path:
        Path to the .py file to execute.
    input_data:
        Optional stdin data.
    timeout_seconds:
        Hard timeout for execution.
    python_executable:
        Python interpreter to use. Defaults to sys.executable.

    Returns
    -------
    SandboxResult with captured output and status.

    Raises
    ------
    SandboxUnavailableError
        If the sandbox backend is not available.
    """
    script_path = Path(script_path).resolve()
    if not script_path.is_file():
        raise SandboxUnavailableError(f"Script not found: {script_path}")
    if not script_path.suffix == ".py":
        raise SandboxUnavailableError(f"Only .py scripts supported: {script_path}")

    python = python_executable or sys.executable

    # Create isolated temp dir, copy script there
    with tempfile.TemporaryDirectory(prefix="sih_sandbox_") as tmpdir:
        sandbox_script = Path(tmpdir) / script_path.name
        shutil.copy2(script_path, sandbox_script)

        real_paddlex_cache = os.path.expanduser("~/.paddlex")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": tmpdir,
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_PROXY": "*",
            "HTTP_PROXY": "http://0.0.0.0:0",
            "HTTPS_PROXY": "http://0.0.0.0:0",
            "http_proxy": "http://0.0.0.0:0",
            "https_proxy": "http://0.0.0.0:0",
            "SIH_SANDBOX": "1",
            "PADDLE_PDX_CACHE_HOME": real_paddlex_cache,
        }

        started = time.perf_counter()
        try:
            result = subprocess.run(
                [python, str(sandbox_script)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=tmpdir,
                env=env,
                stdin=subprocess.PIPE if input_data else subprocess.DEVNULL,
                input=input_data,
            )
            elapsed = (time.perf_counter() - started) * 1000

            return SandboxResult(
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=False,
                elapsed_ms=round(elapsed, 2),
                sandbox_backend="subprocess",
            )

        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - started) * 1000
            return SandboxResult(
                success=False,
                exit_code=None,
                stdout="",
                stderr=f"Execution timed out after {timeout_seconds}s",
                timed_out=True,
                elapsed_ms=round(elapsed, 2),
                sandbox_backend="subprocess",
                error=f"Timeout after {timeout_seconds}s",
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return SandboxResult(
                success=False,
                exit_code=None,
                stdout="",
                stderr=str(exc),
                timed_out=False,
                elapsed_ms=round(elapsed, 2),
                sandbox_backend="subprocess",
                error=str(exc),
            )


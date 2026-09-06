"""Local Docker-backed command sandbox.

This backend is deliberately opt-in. Commands run as container root inside a
network-disabled container; they never receive the host Docker socket, host
root filesystem, or host capabilities. The workspace is the only writable
bind mount and every command is approval-gated by the caller.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


class ContainerSandbox:
    def __init__(self, workspace: str | Path, image: str | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.image = image or os.getenv("AEGIS_CONTAINER_IMAGE", "python:3.12-slim")

    def status(self) -> dict[str, Any]:
        """Return readiness without pulling images or changing Docker state."""
        if not self.workspace.exists() or not self.workspace.is_dir():
            return self._unavailable("Workspace root is unavailable.")
        try:
            daemon = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            if daemon.returncode != 0:
                return self._unavailable((daemon.stderr or "Docker daemon unavailable")[:300])
            image = subprocess.run(
                ["docker", "image", "inspect", self.image, "--format", "{{.Id}}"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            if image.returncode != 0:
                return self._unavailable(
                    f"Container image '{self.image}' is not available locally. "
                    f"Pull or build it before enabling container mode."
                )
            return {
                "ready": True, "status": "ready", "backend": "docker-container",
                "workspace": str(self.workspace), "image": self.image,
                "network": "none", "capabilities": "all-dropped",
                "message": "Docker sandbox is ready.",
            }
        except FileNotFoundError:
            return self._unavailable("Docker CLI is not installed.")
        except Exception as exc:
            return self._unavailable(f"Docker preflight failed: {type(exc).__name__}: {exc}")

    def run(self, command: str, cwd: Path, timeout: int = 90) -> dict[str, Any]:
        """Run a command through ``/bin/sh`` inside the isolated container."""
        relative_cwd = cwd.resolve().relative_to(self.workspace)
        container_cwd = "/workspace" if str(relative_cwd) == "." else f"/workspace/{relative_cwd}"
        docker_command = [
            "docker", "run", "--rm",
            "--network", "none",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "256",
            "--memory", "512m",
            "--cpus", "2",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            # Bind mounts are read-write by default; use key=value fields only.
            "--mount", f"type=bind,src={self.workspace},dst=/workspace",
            "--workdir", container_cwd,
            "--user", "0:0",
            self.image, "/bin/sh", "-lc", command,
        ]
        started = time.perf_counter()
        safe_timeout = max(1, min(120, int(timeout)))
        try:
            proc = subprocess.Popen(
                docker_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True, start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=safe_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                stdout, stderr = proc.communicate()
                return self._result(False, proc.returncode, stdout, stderr,
                                    started, error="command_timeout", timed_out=True,
                                    process_group_terminated=True)
            return self._result(proc.returncode == 0, proc.returncode, stdout, stderr, started)
        except FileNotFoundError:
            return self._result(False, None, "", "Docker CLI is not installed.", started,
                                error="sandbox_unavailable")
        except Exception as exc:
            return self._result(False, None, "", str(exc), started, error=type(exc).__name__)

    def _result(self, ok: bool, exit_code: int | None, stdout: str, stderr: str,
                started: float, error: str | None = None, timed_out: bool = False,
                process_group_terminated: bool = False) -> dict[str, Any]:
        max_len = 12000
        stdout = stdout or ""
        stderr = stderr or ""
        truncated = len(stdout) > max_len or len(stderr) > max_len
        result = {
            "ok": ok, "status": "success" if ok else "failure", "tool": "execute_command",
            "exit_code": exit_code, "stdout": stdout[:max_len], "stderr": stderr[:max_len],
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "truncated": truncated, "timed_out": timed_out,
            "sandbox": {"ready": True, "status": "completed", "backend": "docker-container",
                        "image": self.image, "network": "none", "capabilities": "all-dropped",
                        "process_group": "isolated-container"},
        }
        if error:
            result["error"] = error
        if process_group_terminated:
            result["process_group_terminated"] = True
        return result

    def _unavailable(self, message: str) -> dict[str, Any]:
        return {"ready": False, "status": "unavailable", "backend": "docker-container",
                "workspace": str(self.workspace), "image": self.image, "message": message}

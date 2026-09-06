"""Rootless Bubblewrap command sandbox for Linux hosts."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.cgroup_limits import CgroupLimitError, CgroupV2Limit, cgroup_mode
from tools.sandbox_config import SandboxConfig


class BubblewrapSandbox:
    def __init__(self, workspace: str | Path, config: SandboxConfig | None = None) -> None:
        self.workspace = Path(workspace).resolve()
        self.config = config or SandboxConfig()

    def _base(self, cwd: Path, command: str) -> list[str]:
        relative = cwd.relative_to(self.workspace)
        container_cwd = "/workspace" if str(relative) == "." else f"/workspace/{relative}"
        args = [
            self.config.bwrap_path, "--unshare-user", "--unshare-pid",
            "--share-net" if self.config.network_access else "--unshare-net", "--unshare-ipc",
            "--unshare-uts", "--unshare-cgroup-try", "--disable-userns", "--new-session", "--die-with-parent",
            "--uid", "0", "--gid", "0", "--cap-drop", "ALL", "--clearenv",
            "--setenv", "PATH", "/workspace/.venv/bin:/usr/local/bin:/usr/bin:/bin",
            "--setenv", "HOME", "/tmp/home", "--setenv", "TMPDIR", "/tmp",
            "--setenv", "PYTHONUNBUFFERED", "1",
            "--ro-bind-try", "/usr", "/usr",
            "--ro-bind-try", "/bin", "/bin",
            "--ro-bind-try", "/sbin", "/sbin",
            "--ro-bind-try", "/lib", "/lib",
            "--ro-bind-try", "/lib64", "/lib64",
            "--dir", "/home", "--dir", "/tmp/home", "--dir", "/etc",
            "--ro-bind-try", "/etc/passwd", "/etc/passwd",
            "--ro-bind-try", "/etc/group", "/etc/group",
            "--ro-bind-try", "/etc/nsswitch.conf", "/etc/nsswitch.conf",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(self.workspace), "/workspace", "--chdir", container_cwd,
        ]
        for key in self.config.env_allow:
            if key in os.environ:
                args.extend(["--setenv", key, os.environ[key]])
        for key, value in self.config.extra_env.items():
            args.extend(["--setenv", key, value])
        if self.config.gpu:
            if Path("/dev/dri").is_dir():
                args.extend(["--dev-bind", "/dev/dri", "/dev/dri"])
            for device in sorted(Path("/dev").glob("nvidia*")):
                args.extend(["--dev-bind", str(device), str(device)])
        for host_path, sandbox_path in self.config.extra_ro_binds:
            host = Path(host_path).resolve()
            if host.exists():
                args.extend(["--ro-bind", str(host), sandbox_path])
        args.extend(["--", "/bin/sh", "-lc", command])
        return args

    def dry_run(self, command: str, cwd: Path | None = None) -> dict[str, Any]:
        """Return the exact argv and shell-safe rendering without executing it."""
        target = (cwd or self.workspace).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return {"ok": False, "error": "outside_workspace"}
        argv = self._base(target, self._normalize_command(command))
        import shlex
        return {"ok": True, "tool": "sandbox_dry_run", "argv": argv, "command": shlex.join(argv),
                "network": "shared" if self.config.network_access else "none",
                "writable": ["/workspace", "/tmp"], "resource_limits": {
                    "memory_mb": self.config.memory_limit_mb, "max_pids": self.config.max_pids,
                    "cpu_weight": self.config.cpu_weight}}

    def status(self) -> dict[str, Any]:
        if not self.workspace.is_dir():
            return self._unavailable("Workspace root is unavailable.")
        try:
            result = subprocess.run(
                self._base(self.workspace, "printf AEGIS_BWRAP_READY"),
                capture_output=True, text=True, timeout=5, check=False,
            )
            ready = result.returncode == 0 and result.stdout.strip() == "AEGIS_BWRAP_READY"
            return {
                "ready": ready, "status": "ready" if ready else "unavailable",
                "backend": "bubblewrap", "workspace": str(self.workspace),
                "namespaces": ["user", "pid", "mount", "network", "ipc", "uts"],
                "network": "none", "writable": ["/workspace", "/tmp"],
                "resource_limits": self._resource_limit_status(),
                "message": "Bubblewrap sandbox is ready." if ready else (result.stderr or "probe failed")[:300],
            }
        except FileNotFoundError:
            return self._unavailable("Bubblewrap (bwrap) is not installed.")
        except Exception as exc:
            return self._unavailable(f"Bubblewrap preflight failed: {type(exc).__name__}: {exc}")

    def run(self, command: str, cwd: Path, timeout: int = 90) -> dict[str, Any]:
        command = self._normalize_command(command)
        started = time.perf_counter()
        safe_timeout = max(1, min(int(self.config.timeout_seconds), int(timeout)))
        limiter: CgroupV2Limit | None = None
        try:
            proc = subprocess.Popen(
                self._base(cwd, command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True, start_new_session=True,
            )
            mode = cgroup_mode()
            if mode != "off":
                limiter = CgroupV2Limit(memory_mb=self.config.memory_limit_mb,
                                        max_pids=self.config.max_pids,
                                        cpu_weight=self.config.cpu_weight)
                try:
                    limiter.create()
                    limiter.attach(proc.pid)
                except CgroupLimitError:
                    limiter.destroy()
                    limiter = None
                    if mode == "on":
                        proc.kill()
                        proc.wait()
                        return self._result(False, proc.returncode, "", "cgroup limits unavailable", started,
                                            error="resource_limits_unavailable")
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
                result = self._result(False, proc.returncode, stdout, stderr, started,
                                    error="command_timeout", timed_out=True,
                                    process_group_terminated=True)
                result["sandbox"]["resource_limits"] = bool(limiter)
                return result
            finally:
                if limiter is not None:
                    limiter.destroy()
            result = self._result(proc.returncode == 0, proc.returncode, stdout, stderr, started)
            result["sandbox"]["resource_limits"] = bool(limiter)
            return result
        except FileNotFoundError:
            return self._result(False, None, "", "Bubblewrap is not installed.", started,
                                error="sandbox_unavailable")
        except Exception as exc:
            return self._result(False, None, "", str(exc), started, error=type(exc).__name__)

    def _normalize_command(self, command: str) -> str:
        """Repair host virtualenv launchers after the workspace is remounted.

        A virtualenv's ``pytest`` script often contains a host-absolute
        shebang. The interpreter and packages are present in /workspace, but
        the host shebang path is intentionally hidden by the namespace. Invoke
        the mounted interpreter explicitly instead of exposing host paths.
        """
        # Models commonly echo the host path returned by read_file.  That
        # path is intentionally not mounted in the namespace; translate only
        # the configured workspace prefix to its stable sandbox alias.
        normalized = command.replace(str(self.workspace), "/workspace")
        normalized = re.sub(r"^(\s*)(?:\.venv/)?pytest(?=\s|$)",
                            r"\1/workspace/.venv/bin/python -m pytest", normalized)
        normalized = re.sub(r"^(\s*)\.venv/bin/python(?=\s|$)",
                            r"\1/workspace/.venv/bin/python", normalized)
        return normalized

    def _result(self, ok: bool, exit_code: int | None, stdout: str, stderr: str,
                started: float, error: str | None = None, timed_out: bool = False,
                process_group_terminated: bool = False) -> dict[str, Any]:
        stdout, stderr = stdout or "", stderr or ""
        max_len = self.config.max_output_bytes
        truncated = len(stdout) > max_len or len(stderr) > max_len
        result = {
            "ok": ok, "status": "success" if ok else "failure", "tool": "execute_command",
            "exit_code": exit_code, "stdout": stdout[:max_len], "stderr": stderr[:max_len],
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            "truncated": truncated, "timed_out": timed_out,
            "sandbox": {"ready": True, "status": "completed", "backend": "bubblewrap",
                "network": "shared" if self.config.network_access else "none", "process_group": "isolated",
                "writable": ["/workspace", "/tmp"], "resource_limits": bool(self.config)},
        }
        if error:
            result["error"] = error
        if process_group_terminated:
            result["process_group_terminated"] = True
        return result

    def _unavailable(self, message: str) -> dict[str, Any]:
        return {"ready": False, "status": "unavailable", "backend": "bubblewrap",
                "workspace": str(self.workspace), "message": message}

    def _resource_limit_status(self) -> dict[str, Any]:
        mode = cgroup_mode()
        available = CgroupV2Limit().available if mode != "off" else False
        return {"mode": mode, "available": available, "configured": False}

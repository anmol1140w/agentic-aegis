"""Persistent native sandbox API for bounded agent workflows.

This is the AEGIS equivalent of a small DeepAgents-style sandbox object: a
fresh workspace can persist across several commands, files can be transferred
through validated relative paths, and the underlying command always uses the
AEGIS Bubblewrap backend.
"""

from __future__ import annotations

import asyncio
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.bwrap_sandbox import BubblewrapSandbox
from tools.sandbox_config import SandboxConfig


@dataclass(frozen=True)
class NativeExecuteResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False
    oom_killed: bool = False


class NativeSandbox:
    def __init__(self, config: SandboxConfig | None = None, *, root: str | Path | None = None) -> None:
        self.config = config or SandboxConfig()
        self._provided_root = Path(root).resolve() if root is not None else None
        self.root: Path | None = None
        self._temporary_root: tempfile.TemporaryDirectory[str] | None = None
        self._backend: BubblewrapSandbox | None = None

    def __enter__(self) -> "NativeSandbox":
        if self.root is not None:
            return self
        if self._provided_root is not None:
            self._provided_root.mkdir(parents=True, exist_ok=True)
            self.root = self._provided_root
        else:
            self._temporary_root = tempfile.TemporaryDirectory(prefix="aegis-sandbox-")
            self.root = Path(self._temporary_root.name)
        self._backend = BubblewrapSandbox(self.root, self.config)
        return self

    async def __aenter__(self) -> "NativeSandbox":
        return self.__enter__()

    def execute(self, command: str, *, cwd: str = ".", timeout: float | None = None) -> NativeExecuteResult:
        backend = self._require_backend()
        target = self._resolve(cwd)
        result = backend.run(command, target, timeout=int(timeout or self.config.timeout_seconds))
        return NativeExecuteResult(result.get("exit_code"), result.get("stdout", ""), result.get("stderr", ""),
                                   result.get("timed_out", False), result.get("truncated", False),
                                   result.get("exit_code") == 137)

    async def aexecute(self, command: str, *, cwd: str = ".", timeout: float | None = None) -> NativeExecuteResult:
        return await asyncio.to_thread(self.execute, command, cwd=cwd, timeout=timeout)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        root = self._require_root()
        results = []
        for relative, content in files:
            try:
                target = self._resolve(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                results.append({"path": relative, "success": True})
            except (ValueError, OSError) as exc:
                results.append({"path": relative, "success": False, "error": str(exc)[:200]})
        return results

    def download_files(self, paths: list[str]) -> list[dict[str, Any]]:
        self._require_root()
        results = []
        for relative in paths:
            try:
                target = self._resolve(relative)
                results.append({"path": relative, "success": True, "content": target.read_bytes()})
            except (ValueError, OSError) as exc:
                results.append({"path": relative, "success": False, "error": str(exc)[:200]})
        return results

    def list_files(self, prefix: str = ".") -> list[str]:
        target = self._resolve(prefix)
        return sorted(str(p.relative_to(self._require_root())) for p in target.rglob("*") if p.is_file())

    def dry_run(self, command: str, *, cwd: str = ".") -> dict[str, Any]:
        backend = self._require_backend()
        return backend.dry_run(command, self._resolve(cwd))

    def close(self) -> None:
        self._backend = None
        self.root = None
        if self._temporary_root is not None:
            self._temporary_root.cleanup()
            self._temporary_root = None

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await asyncio.to_thread(self.close)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _require_root(self) -> Path:
        if self.root is None:
            raise RuntimeError("Sandbox not initialized; use it as a context manager")
        return self.root

    def _require_backend(self) -> BubblewrapSandbox:
        if self._backend is None:
            raise RuntimeError("Sandbox not initialized; use it as a context manager")
        return self._backend

    def _resolve(self, relative: str) -> Path:
        if not relative or Path(relative).is_absolute():
            raise ValueError("path must be relative to the sandbox workspace")
        root = self._require_root()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError("path escapes sandbox workspace")
        return target

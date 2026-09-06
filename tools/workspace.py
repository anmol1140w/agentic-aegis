"""Bounded, read-only tools rooted in the controlled local workspace."""

from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from runtime.command_policy import evaluate_command

_MAX_ITEMS = 200
_MAX_FILE_BYTES = 64 * 1024
_MAX_SEARCH_RESULTS = 100


class WorkspaceReadTools:
    """Read-only coding-agent tools; every resolved path must remain under ``root``."""

    def __init__(self, root: str | Path = "workspace", approver: Any = None, command_approver: Any = None) -> None:
        self.root = Path(root).resolve()
        self.approver = approver
        self.command_approver = command_approver or approver

    def _path(self, raw: str | Path) -> Path | dict[str, Any]:
        target = (self.root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            target.relative_to(self.root)
        except ValueError:
            return {"ok": False, "error": "OutsideWorkspace", "message": "Path is outside the controlled workspace."}
        return target

    @staticmethod
    def _item(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {"name": path.name, "path": str(path), "type": "directory" if path.is_dir() else "file", "size": stat.st_size if path.is_file() else None, "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}

    def list_directory(self, path: str = ".") -> dict[str, Any]:
        target = self._path(path)
        if isinstance(target, dict): return target
        if not target.is_dir(): return {"ok": False, "error": "NotDirectory"}
        items = sorted(target.iterdir(), key=lambda item: item.name.lower())[:_MAX_ITEMS]
        return {"ok": True, "tool": "list_directory", "path": str(target), "items": [self._item(item) for item in items], "truncated": len(items) == _MAX_ITEMS}

    def read_file(self, path: str) -> dict[str, Any]:
        target = self._path(path)
        if isinstance(target, dict): return target
        if not target.is_file(): return {"ok": False, "error": "NotFile"}
        if target.stat().st_size > _MAX_FILE_BYTES: return {"ok": False, "error": "FileTooLarge"}
        return {"ok": True, "tool": "read_file", "path": str(target), "content": target.read_text(encoding="utf-8", errors="replace")}

    def find_files(self, pattern: str, path: str = ".") -> dict[str, Any]:
        root = self._path(path)
        if isinstance(root, dict): return root
        if not root.is_dir(): return {"ok": False, "error": "NotDirectory"}
        matches = [self._item(item) for item in root.rglob("*") if item.is_file() and fnmatch.fnmatch(item.name, pattern)][: _MAX_SEARCH_RESULTS]
        return {"ok": True, "tool": "find_files", "items": matches, "truncated": len(matches) == _MAX_SEARCH_RESULTS}

    def search_files(self, query: str, path: str = ".") -> dict[str, Any]:
        root = self._path(path)
        if isinstance(root, dict): return root
        matches: list[dict[str, Any]] = []
        for item in root.rglob("*"):
            if len(matches) >= _MAX_SEARCH_RESULTS: break
            if not item.is_file() or item.stat().st_size > _MAX_FILE_BYTES: continue
            try: lines = item.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError: continue
            for line_no, line in enumerate(lines, 1):
                if query in line:
                    matches.append({"path": str(item), "line": line_no, "text": line[:500]})
                    if len(matches) >= _MAX_SEARCH_RESULTS: break
        return {"ok": True, "tool": "search_files", "matches": matches, "truncated": len(matches) == _MAX_SEARCH_RESULTS}

    def get_file_info(self, path: str) -> dict[str, Any]:
        target = self._path(path)
        return target if isinstance(target, dict) else ({"ok": True, "tool": "get_file_info", "item": self._item(target)} if target.exists() else {"ok": False, "error": "NotFound"})

    def _git(self, *args: str) -> dict[str, Any]:
        try:
            result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, timeout=10, check=False)
        except OSError as exc:
            return {"ok": False, "error": "GitUnavailable", "message": str(exc)}
        return {"ok": result.returncode == 0, "tool": f"git_{args[0]}", "output": (result.stdout or result.stderr)[:_MAX_FILE_BYTES]}

    def git_status(self) -> dict[str, Any]: return self._git("status", "--short")
    def git_diff(self) -> dict[str, Any]: return self._git("diff", "--stat")

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        approver: Any = None,
    ) -> dict[str, Any]:
        """Atomically edit a file in the workspace by replacing single exact old_text with new_text.

        Requires approval. Validates that the path is strictly within workspace root.
        """
        raw_p = Path(path)
        if raw_p.is_absolute():
            candidate = raw_p.resolve()
        else:
            candidate = (self.root / raw_p).resolve()

        # Reject path if outside workspace root
        try:
            rel = candidate.relative_to(self.root)
            if candidate == self.root:
                raise ValueError("Target cannot be workspace root directory.")
        except (ValueError, RuntimeError):
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "outside_workspace",
                "message": "Path is outside the controlled workspace.",
                "path": str(path),
            }

        rel_str = str(rel)

        # Check existence and regular file
        if not candidate.exists():
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "file_not_found",
                "message": f"File does not exist: {path}",
                "path": rel_str,
            }
        if not candidate.is_file():
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "file_not_found",
                "message": f"Path is not a regular file: {path}",
                "path": rel_str,
            }

        # Check file size limit
        if candidate.stat().st_size > _MAX_FILE_BYTES:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "file_too_large",
                "message": f"File exceeds {_MAX_FILE_BYTES} bytes limit.",
                "path": rel_str,
            }
        if len(new_text.encode("utf-8")) > _MAX_FILE_BYTES:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "file_too_large",
                "message": f"Replacement exceeds {_MAX_FILE_BYTES} bytes limit.",
                "path": rel_str,
            }

        if not old_text:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "empty_old_text",
                "message": "old_text cannot be empty.",
                "path": rel_str,
            }

        if old_text == new_text:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "identical_replacement",
                "message": "old_text and new_text are identical.",
                "path": rel_str,
            }

        try:
            raw_bytes = candidate.read_bytes()
            if b"\x00" in raw_bytes[:4096]:
                return {
                    "ok": False,
                    "status": "failure",
                    "tool": "edit_file",
                    "error": "binary_file",
                    "message": "Cannot edit binary file.",
                    "path": rel_str,
                }
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "binary_file",
                "message": "Cannot edit non-UTF-8 binary file.",
                "path": rel_str,
            }
        except PermissionError as exc:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "permission_denied",
                "message": str(exc),
                "path": rel_str,
            }
        except OSError as exc:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "file_not_found",
                "message": str(exc),
                "path": rel_str,
            }

        # Validate old_text matches exactly once
        count = content.count(old_text)
        if count == 0:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "old_text_not_found",
                "message": "old_text was not found in the file.",
                "path": rel_str,
            }
        if count > 1:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "old_text_ambiguous",
                "message": f"old_text matched {count} times, expected exactly 1.",
                "path": rel_str,
            }

        # Approval gate
        active_approver = approver if approver is not None else getattr(self, "approver", None)
        approved = False
        if active_approver is not None:
            approved = self._check_approval(active_approver, rel_str, old_text, new_text)

        if not approved:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "approval_denied",
                "message": "Approval denied for edit_file.",
                "path": rel_str,
            }

        # Atomic replace
        import tempfile
        bytes_before = len(content.encode("utf-8"))
        new_content = content.replace(old_text, new_text, 1)
        bytes_after = len(new_content.encode("utf-8"))

        try:
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(candidate.parent),
                encoding="utf-8",
                delete=False,
            )
            temp_file.write(new_content)
            temp_file.flush()
            temp_file.close()
            Path(temp_file.name).replace(candidate)
            # Invalidate stale bytecode in workspace if any exists
            pycache_dir = candidate.parent / "__pycache__"
            if pycache_dir.is_dir():
                for pyc in pycache_dir.glob(f"{candidate.stem}.*.pyc"):
                    try:
                        pyc.unlink(missing_ok=True)
                    except OSError:
                        pass
        except OSError as exc:
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": "permission_denied",
                "message": f"Failed to write file atomically: {exc}",
                "path": rel_str,
            }

        return {
            "ok": True,
            "status": "success",
            "tool": "edit_file",
            "path": rel_str,
            "changed": True,
            "bytes_before": bytes_before,
            "bytes_after": bytes_after,
        }

    def create_file(self, path: str, content: str, approver: Any = None) -> dict[str, Any]:
        """Create a new text file inside the workspace (requires approval)."""
        target = self._path(path)
        if isinstance(target, dict):
            return target
        if target.exists():
            return {"ok": False, "status": "failure", "tool": "create_file", "error": "file_exists"}
        if not isinstance(content, str) or len(content.encode("utf-8")) > _MAX_FILE_BYTES:
            return {"ok": False, "status": "failure", "tool": "create_file", "error": "content_too_large"}
        active = approver if approver is not None else self.approver
        approved = self._check_create_approval(active, path, content) if active else False
        if not approved:
            return {"ok": False, "status": "failure", "tool": "create_file", "error": "approval_denied"}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return {"ok": True, "status": "success", "tool": "create_file", "path": str(target),
                    "bytes": target.stat().st_size}
        except OSError as exc:
            return {"ok": False, "status": "failure", "tool": "create_file", "error": str(exc)[:200]}

    def create_python_script(self, path: str, content: str, approver: Any = None) -> dict[str, Any]:
        """Create one approved Python source file inside the workspace.

        This is intentionally a narrow convenience tool for coding tasks; it
        does not execute the script or broaden filesystem permissions.
        """
        if not str(path).lower().endswith(".py"):
            return {"ok": False, "status": "failure", "tool": "create_python_script",
                    "error": "invalid_python_path", "message": "Path must end with .py."}
        result = self.create_file(path, content, approver=approver)
        result["tool"] = "create_python_script"
        return result

    def execute_command(
        self,
        command: str,
        cwd: str = ".",
        timeout: int = 30,
        approver: Any = None,
    ) -> dict[str, Any]:
        """Execute a controlled command inside the workspace root under strict policy."""
        raw_cwd = Path(cwd) if cwd else Path(".")
        target_cwd = (self.root / raw_cwd).resolve() if not raw_cwd.is_absolute() else raw_cwd.resolve()

        # Validate canonical cwd within workspace root
        try:
            rel_cwd = target_cwd.relative_to(self.root)
        except ValueError:
            return {
                "ok": False,
                "status": "failure",
                "tool": "execute_command",
                "error": "outside_workspace",
                "message": "cwd is outside the controlled workspace.",
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "duration_ms": 0,
                "truncated": False,
            }

        if not target_cwd.exists() or not target_cwd.is_dir():
            return {
                "ok": False,
                "status": "failure",
                "tool": "execute_command",
                "error": "directory_not_found",
                "message": f"Working directory does not exist: {cwd}",
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "duration_ms": 0,
                "truncated": False,
            }

        decision = evaluate_command(command, target_cwd, self.root)
        if not decision.allowed:
            return {
                "ok": False,
                "status": "failure",
                "tool": "execute_command",
                "error": "disallowed_command",
                "message": decision.reason,
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "duration_ms": 0,
                "truncated": False,
            }

        if decision.requires_approval:
            active_approver = approver if approver is not None else getattr(self, "command_approver", getattr(self, "approver", None))
            approved = False
            if active_approver is not None:
                approved = self._check_command_approval(active_approver, command, str(rel_cwd))
            if not approved:
                return {
                    "ok": False,
                    "status": "failure",
                    "tool": "execute_command",
                    "error": "approval_denied",
                    "message": "Approval denied for command execution.",
                    "exit_code": None,
                    "stdout": "",
                    "stderr": "",
                    "duration_ms": 0,
                    "truncated": False,
                }

        # Setup restricted environment
        safe_timeout = max(1, min(60, int(timeout)))
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(self.root),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        # Provide access to local venv python and workspace root for pytest
        project_root = Path(__file__).resolve().parent.parent
        venv_dir = self.root / ".venv"
        if not venv_dir.exists() and (self.root.parent / ".venv").exists():
            venv_dir = (self.root.parent / ".venv").resolve()
        if not venv_dir.exists() and (project_root / ".venv").exists():
            venv_dir = (project_root / ".venv").resolve()
        if venv_dir.exists():
            env["VIRTUAL_ENV"] = str(venv_dir)
            env["PATH"] = f"{venv_dir / 'bin'}:{env['PATH']}"
        env["PYTHONPATH"] = str(self.root)

        tokens = shlex.split(command)
        if tokens:
            if tokens[0].startswith(".venv/"):
                if (target_cwd / tokens[0]).exists():
                    tokens[0] = str(target_cwd / tokens[0])
                elif (venv_dir / tokens[0].removeprefix(".venv/")).exists():
                    tokens[0] = str(venv_dir / tokens[0].removeprefix(".venv/"))
                elif (project_root / tokens[0]).exists():
                    tokens[0] = str(project_root / tokens[0])

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                tokens,
                cwd=str(target_cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=safe_timeout,
                check=False,
            )
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            raw_out = getattr(exc, "output", None) or getattr(exc, "stdout", "")
            raw_err = getattr(exc, "stderr", "")
            out = raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, bytes) else str(raw_out or "")
            err = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, bytes) else str(raw_err or "")
            return {
                "ok": False,
                "status": "failure",
                "tool": "execute_command",
                "error": "command_timeout",
                "message": f"Command timed out after {safe_timeout} seconds",
                "exit_code": None,
                "stdout": out[:12000],
                "stderr": err[:12000],
                "duration_ms": duration_ms,
                "truncated": False,
            }
        except Exception as exc:
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "ok": False,
                "status": "failure",
                "tool": "execute_command",
                "error": type(exc).__name__,
                "message": str(exc),
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "duration_ms": duration_ms,
                "truncated": False,
            }

        max_len = 12000
        truncated = False
        if len(stdout) > max_len:
            stdout = stdout[:max_len]
            truncated = True
        if len(stderr) > max_len:
            stderr = stderr[:max_len]
            truncated = True

        success = (exit_code == 0)
        res = {
            "ok": success,
            "status": "success" if success else "failure",
            "tool": "execute_command",
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "duration_ms": duration_ms,
            "truncated": truncated,
        }
        if decision.requires_approval:
            res["approval"] = "approved"
        return res

    @staticmethod
    def _check_command_approval(approver: Any, command: str, cwd: str) -> bool:
        """Call approver callback for command execution."""
        summary = f"Execute command in '{cwd}': {command}"
        try:
            return bool(approver("execute_command", command, cwd))
        except TypeError:
            pass
        try:
            return bool(approver("execute_command", summary))
        except TypeError:
            pass
        try:
            return bool(approver())
        except TypeError:
            return False

    @staticmethod
    def _check_create_approval(approver: Any, path: str, content: str) -> bool:
        """Call create approval callbacks while keeping content bounded."""
        summary = f"Create file '{path}' ({len(content)} chars)"
        for args in (("create_file", path, content, ""), ("create_file", summary), ()): 
            try:
                return bool(approver(*args))
            except TypeError:
                continue
        return False

    @staticmethod
    def _check_approval(approver: Any, path: str, old_text: str, new_text: str) -> bool:
        """Call approver callback tolerating different signatures."""
        summary = f"Edit {path}: replace {len(old_text)} chars with {len(new_text)} chars"
        try:
            return bool(approver("edit_file", path, old_text, new_text))
        except TypeError:
            pass
        try:
            return bool(approver("edit_file", summary))
        except TypeError:
            pass
        try:
            return bool(approver("edit_task", "edit_file", summary))
        except TypeError:
            pass
        try:
            return bool(approver())
        except TypeError:
            return False

    def as_langchain_tools(self) -> list[BaseTool]:
        """Expose only workspace-rooted schemas to model tool loops."""
        @tool
        def list_directory(path: str = ".") -> dict[str, Any]:
            """List a controlled workspace directory (maximum 200 entries)."""
            return self.list_directory(path)
        @tool
        def read_file(path: str) -> dict[str, Any]:
            """Read and analyze one controlled workspace text file (maximum 64 KiB). Always call this first to inspect and analyze the file before proposing edits."""
            return self.read_file(path)
        @tool
        def search_files(query: str, path: str = ".") -> dict[str, Any]:
            """Search text in controlled workspace files (maximum 100 matches)."""
            return self.search_files(query, path)
        @tool
        def find_files(pattern: str, path: str = ".") -> dict[str, Any]:
            """Find files by glob under the controlled workspace."""
            return self.find_files(pattern, path)
        @tool
        def get_file_info(path: str) -> dict[str, Any]:
            """Get metadata for one controlled workspace path."""
            return self.get_file_info(path)
        @tool
        def git_status() -> dict[str, Any]:
            """Return read-only git status for the controlled workspace."""
            return self.git_status()
        @tool
        def git_diff() -> dict[str, Any]:
            """Return read-only git diff summary for the controlled workspace."""
            return self.git_diff()
        @tool
        def edit_file(path: str, old_text: str, new_text: str) -> dict[str, Any]:
            """Edit a workspace file by replacing exact single old_text with new_text (requires approval). You must read and analyze the file with read_file first before editing to verify the exact old_text."""
            return self.edit_file(path, old_text, new_text)
        @tool
        def create_file(path: str, content: str) -> dict[str, Any]:
            """Create a new workspace text file (requires approval)."""
            return self.create_file(path, content)
        @tool
        def create_python_script(path: str, content: str) -> dict[str, Any]:
            """Create an approved Python script in the workspace; does not execute it."""
            return self.create_python_script(path, content)
        @tool
        def execute_command(command: str, cwd: str = ".", timeout: int = 30) -> dict[str, Any]:
            """Execute an allowed command (e.g. pytest) inside workspace root."""
            return self.execute_command(command, cwd=cwd, timeout=timeout)

        return [list_directory, read_file, search_files, find_files, get_file_info, git_status, git_diff, edit_file, create_file, create_python_script, execute_command]

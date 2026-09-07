"""Bounded, read-only tools rooted in the controlled local workspace."""

from __future__ import annotations

import fnmatch
import ast
import json
import os
import shlex
import subprocess
import signal
import sys
import time
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from runtime.command_policy import evaluate_command
from runtime.capability_policy import CommandCapability, classify_command
from tools.container_sandbox import ContainerSandbox
from tools.bwrap_sandbox import BubblewrapSandbox
from tools import native_process
from storage.checkpoints import WorkspaceCheckpointStore
from runtime.skills import SkillCatalog
from tools.runtime import ToolRuntime

_MAX_ITEMS = 200
_MAX_FILE_BYTES = 64 * 1024
_MAX_SEARCH_RESULTS = 100
_MAX_CONTEXT_FILES = 160
_MAX_CONTEXT_FILE_BYTES = 8 * 1024
_CONTEXT_SKIP_DIRS = {".git", ".venv", ".aegis", "node_modules", "__pycache__", ".pytest_cache", ".tox", ".mypy_cache", "outputs", "temporary", "wheelhouse", "dist"}
_TREE_SKIP_DIRS = _CONTEXT_SKIP_DIRS | {".idea", ".vscode", "build", "coverage", "htmlcov"}


class WorkspaceReadTools:
    """Read-only coding-agent tools; every resolved path must remain under ``root``."""

    def __init__(self, root: str | Path = "workspace", approver: Any = None, command_approver: Any = None) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.execution_dir = self.root / "executions"
        self.artifact_dir = self.root / "artifacts"
        self.approver = approver
        self.command_approver = command_approver or approver
        self.command_event_callback: Any = None
        self.sandbox_backend = os.getenv("AEGIS_SANDBOX_BACKEND", "process").strip().lower()
        self.container_sandbox = ContainerSandbox(self.root) if self.sandbox_backend in {"docker", "container"} else None
        self.bwrap_sandbox = BubblewrapSandbox(self.root) if self.sandbox_backend in {"bwrap", "bubblewrap"} else None
        self.checkpoints = WorkspaceCheckpointStore(self.root)
        self.skills = SkillCatalog([self.root / "skills", self.root / ".aegis" / "skills"])
        self.runtime = ToolRuntime(self)

    def tool_catalog(self) -> list[dict[str, Any]]:
        """Return the compact namespaced catalog for progressive disclosure."""
        return self.runtime.catalog()

    def invoke_tool(self, name: str, arguments: dict[str, Any] | None = None,
                    *, request_id: str | None = None) -> dict[str, Any]:
        """Invoke a canonical tool through the single policy/result boundary."""
        return self.runtime.invoke(name, arguments, request_id=request_id)

    def _path(self, raw: str | Path) -> Path | dict[str, Any]:
        raw_path = Path(raw)
        # User-facing paths may be written as workspace/foo.py even when the
        # configured tool root is already .../workspace. Resolve that alias
        # inside the configured workspace; never fall back to repository root.
        if not raw_path.is_absolute() and raw_path.parts and raw_path.parts[0] == self.root.name:
            raw_path = Path(*raw_path.parts[1:]) or Path(".")
        target = (self.root / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
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

    def tree(self, path: str = ".", max_depth: int = 8, max_entries: int = 1000,
             include_hidden: bool = False, include_generated: bool = False) -> dict[str, Any]:
        """Return a bounded recursive tree without reading file contents."""
        target = self._path(path)
        if isinstance(target, dict):
            return target
        if not target.is_dir():
            return {"ok": False, "tool": "tree", "error": "NotDirectory", "path": str(path)}
        depth_limit = max(0, min(32, int(max_depth)))
        entry_limit = max(1, min(10_000, int(max_entries)))
        skipped = set() if include_generated else _TREE_SKIP_DIRS
        entries: list[dict[str, Any]] = []
        lines = [target.name or str(target)]
        truncated = False

        def visit(directory: Path, depth: int, prefix: str) -> None:
            nonlocal truncated
            if truncated:
                return
            try:
                children = sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
            except OSError:
                return
            visible = [child for child in children
                       if (include_hidden or not child.name.startswith("."))
                       and (not child.is_dir() or child.name not in skipped)]
            for index, child in enumerate(visible):
                if len(entries) >= entry_limit:
                    truncated = True
                    lines.append(prefix + "└── … (entry limit reached)")
                    return
                is_dir = child.is_dir()
                rel = str(child.relative_to(self.root))
                entries.append({"path": rel, "name": child.name,
                                "type": "directory" if is_dir else "file",
                                "depth": depth + 1})
                branch = "└── " if index == len(visible) - 1 else "├── "
                lines.append(prefix + branch + child.name + ("/" if is_dir else ""))
                if is_dir and depth < depth_limit:
                    visit(child, depth + 1, prefix + ("    " if index == len(visible) - 1 else "│   "))
                elif is_dir and depth >= depth_limit:
                    entries[-1]["truncated"] = True

        visit(target, 0, "")
        rendered = "\n".join(lines)
        return {"ok": True, "tool": "tree", "path": str(target), "entries": entries,
                "tree": rendered[:_MAX_FILE_BYTES], "entry_count": len(entries),
                "max_depth": depth_limit, "truncated": truncated or len(rendered) > _MAX_FILE_BYTES}

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

    def repository_context(self) -> dict[str, Any]:
        """Return bounded repository facts for the first coding-agent decision.

        This is the terminal equivalent of the reference projects' workspace
        context gathering: deterministic structure and a few bounded contract
        files are supplied as evidence before the model starts reasoning.
        It never reads outside ``root`` and never executes repository code.
        """
        root = self.root
        if not root.exists() or not root.is_dir():
            return {"ok": False, "tool": "repository_context", "error": "workspace_unavailable"}
        try:
            top_level = [self._item(item) for item in sorted(root.iterdir(), key=lambda p: p.name.lower())[:_MAX_ITEMS]]
        except OSError as exc:
            return {"ok": False, "tool": "repository_context", "error": "filesystem_error", "message": str(exc)[:300]}

        files: list[str] = []
        extensions: dict[str, int] = {}
        important_names = {"AGENTS.md", "README.md", "README", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile", "Dockerfile"}
        important: dict[str, str] = {}
        for current, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in _CONTEXT_SKIP_DIRS and not d.startswith("."))
            for name in sorted(names):
                path = Path(current) / name
                try:
                    relative = path.relative_to(root)
                except ValueError:
                    continue
                if name in important_names and len(important) < 8:
                    try:
                        important[str(relative)] = path.read_text(encoding="utf-8", errors="replace")[:_MAX_CONTEXT_FILE_BYTES]
                    except OSError:
                        pass
                if len(files) >= _MAX_CONTEXT_FILES:
                    break
                files.append(str(relative))
                suffix = path.suffix.lower() or "[no extension]"
                extensions[suffix] = extensions.get(suffix, 0) + 1
            if len(files) >= _MAX_CONTEXT_FILES:
                break

        git = self._git("status", "--short") if (root / ".git").exists() else {"ok": True, "output": "not a git repository"}
        return {"ok": True, "tool": "repository_context", "root": str(root),
                "top_level": top_level, "files": files, "file_count": len(files),
                "truncated": len(files) >= _MAX_CONTEXT_FILES,
                "extensions": dict(sorted(extensions.items(), key=lambda item: (-item[1], item[0]))),
                "important_files": important, "git_status": git,
                "skills": self.skills.list()}

    def list_skills(self) -> dict[str, Any]:
        return {"ok": True, "tool": "list_skills", "skills": self.skills.list()}

    def read_skill(self, name: str) -> dict[str, Any]:
        try:
            return {"ok": True, "tool": "read_skill", **self.skills.load(name)}
        except KeyError as exc:
            return {"ok": False, "tool": "read_skill", "error": "skill_not_found", "message": str(exc)}

    def _git(self, *args: str) -> dict[str, Any]:
        try:
            result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True, timeout=70, check=False)
        except OSError as exc:
            return {"ok": False, "error": "GitUnavailable", "message": str(exc)}
        return {"ok": result.returncode == 0, "tool": f"git_{args[0]}", "output": (result.stdout or result.stderr)[:_MAX_FILE_BYTES]}

    def git_status(self) -> dict[str, Any]: return self._git("status", "--short")
    def git_diff(self) -> dict[str, Any]: return self._git("diff", "--stat")

    def workspace_diff(self, checkpoint_id: str) -> dict[str, Any]:
        """Return reviewable file changes since a workspace checkpoint."""
        try:
            return {"ok": True, "tool": "workspace_diff", **self.checkpoints.diff(checkpoint_id)}
        except (FileNotFoundError, ValueError) as exc:
            return {"ok": False, "tool": "workspace_diff", "error": "checkpoint_not_found", "message": str(exc)[:200]}

    def create_checkpoint(self, label: str = "checkpoint", approver: Any = None) -> dict[str, Any]:
        """Create an approved workspace snapshot for review or rollback."""
        active = approver if approver is not None else self.approver
        if not active or not self._check_create_approval(active, ".aegis/checkpoints", label):
            return {"ok": False, "tool": "create_checkpoint", "error": "approval_denied"}
        try:
            return {"ok": True, "tool": "create_checkpoint", **self.checkpoints.create(label)}
        except OSError as exc:
            return {"ok": False, "tool": "create_checkpoint", "error": "checkpoint_failed", "message": str(exc)[:200]}

    def list_checkpoints(self) -> dict[str, Any]:
        return {"ok": True, "tool": "list_checkpoints", "checkpoints": self.checkpoints.list()}

    def restore_checkpoint(self, checkpoint_id: str, approver: Any = None) -> dict[str, Any]:
        """Restore a checkpoint; always requires explicit approval."""
        active = approver if approver is not None else self.approver
        if not active or not self._check_command_approval(active, f"restore checkpoint {checkpoint_id}", "."):
            return {"ok": False, "tool": "restore_checkpoint", "error": "approval_denied"}
        try:
            return {"ok": True, "tool": "restore_checkpoint", **self.checkpoints.restore(checkpoint_id)}
        except (FileNotFoundError, ValueError, OSError) as exc:
            return {"ok": False, "tool": "restore_checkpoint", "error": "restore_failed", "message": str(exc)[:200]}

    def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        approver: Any = None,
        expected_matches: int = 1,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Atomically edit a file in the workspace by replacing single exact old_text with new_text.

        Requires approval. Validates that the path is strictly within workspace root.
        """
        raw_p = Path(path)
        if not raw_p.is_absolute() and raw_p.parts and raw_p.parts[0] == self.root.name:
            raw_p = Path(*raw_p.parts[1:]) or Path(".")
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
        if count != expected_matches:
            error = ("old_text_not_found" if expected_matches == 1 and count == 0
                     else "old_text_ambiguous" if expected_matches == 1 and count > 1
                     else "invalid_input")
            return {
                "ok": False,
                "status": "failure",
                "tool": "edit_file",
                "error": error,
                "message": f"Expected {expected_matches} matches, found {count}.",
                "path": rel_str,
            }
        new_content = content.replace(old_text, new_text, expected_matches)
        import difflib
        diff = "".join(difflib.unified_diff(content.splitlines(True), new_content.splitlines(True),
                                             fromfile=rel_str, tofile=rel_str))[:_MAX_FILE_BYTES]
        if dry_run:
            return {"ok": True, "status": "success", "tool": "edit_file", "path": rel_str,
                    "changed": new_content != content, "matches": count, "diff": diff,
                    "dry_run": True}

        # Approval is requested only after all deterministic validation and
        # preflight checks pass. A dry-run must never trigger a human prompt.
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
        new_content = content.replace(old_text, new_text, expected_matches)
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
            "matches": count,
            "diff": diff,
        }

    def create_file(self, path: str, content: str, approver: Any = None,
                    dry_run: bool = False) -> dict[str, Any]:
        """Create a new text file inside the workspace (requires approval)."""
        target = self._path(path)
        if isinstance(target, dict):
            return target
        if target.exists():
            return {"ok": False, "status": "failure", "tool": "create_file", "error": "file_exists"}
        if not isinstance(content, str) or len(content.encode("utf-8")) > _MAX_FILE_BYTES:
            return {"ok": False, "status": "failure", "tool": "create_file", "error": "content_too_large"}
        if dry_run:
            return {"ok": True, "status": "success", "tool": "create_file", "path": str(target),
                    "bytes": len(content.encode("utf-8")), "dry_run": True, "changed": not target.exists()}
        active = approver if approver is not None else self.approver
        approved = self._check_create_approval(active, path, content) if active else False
        if not approved:
            return {"ok": False, "status": "failure", "tool": "create_file", "error": "approval_denied"}
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            result = {"ok": True, "status": "success", "tool": "create_file", "path": str(target),
                      "bytes": target.stat().st_size}
            # Files deliberately created under artifacts/ receive provenance
            # immediately, so the directory is a usable artifact catalog and
            # not just an output bucket.
            if target.is_relative_to(self.artifact_dir) and target.name != "artifacts.json":
                from storage.artifacts import ArtifactManager
                result["artifact"] = ArtifactManager(self.artifact_dir).register_artifact(
                    target, artifact_type="file", tool_id="create_file",
                    verification_status="verified",
                )
            return result
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
        try:
            ast.parse(content, filename=str(path))
        except SyntaxError as exc:
            return {"ok": False, "status": "failure", "tool": "create_python_script",
                    "error": "syntax_error", "message": f"Python syntax error: {exc.msg} (line {exc.lineno})"}
        result = self.create_file(path, content, approver=approver)
        result["tool"] = "create_python_script"
        return result

    def execute_command(
        self,
        command: str,
        cwd: str = ".",
        timeout: int = 90,
        approver: Any = None,
    ) -> dict[str, Any]:
        """Execute a controlled command inside the workspace root under strict policy."""
        sandbox = self.sandbox_status()
        if not sandbox.get("ready"):
            return {
                "ok": False, "status": "failure", "tool": "execute_command",
                "error": "sandbox_unavailable", "message": sandbox.get("message", "Sandbox unavailable"),
                "exit_code": None, "stdout": "", "stderr": "", "duration_ms": 0,
                "truncated": False, "sandbox": sandbox,
            }
        raw_cwd = Path(cwd) if cwd else Path(".")
        if not raw_cwd.is_absolute() and raw_cwd.parts and raw_cwd.parts[0] == self.root.name:
            raw_cwd = Path(*raw_cwd.parts[1:]) or Path(".")
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

        if self.container_sandbox is not None or self.bwrap_sandbox is not None:
            active_approver = approver if approver is not None else getattr(self, "command_approver", getattr(self, "approver", None))
            approved = False
            if active_approver is not None:
                approved = self._check_command_approval(active_approver, command, str(rel_cwd))
            capability = classify_command(command)
            if capability in {CommandCapability.PRIVILEGED, CommandCapability.BLOCKED}:
                return {
                    "ok": False, "status": "failure", "tool": "execute_command",
                    "error": "capability_denied", "message": f"Command capability '{capability.value}' is not available to the agent.",
                    "capability": capability.value, "exit_code": None, "stdout": "", "stderr": "",
                    "duration_ms": 0, "truncated": False,
                    "sandbox": self.sandbox_status(),
                }
            if capability == CommandCapability.NETWORK and os.getenv("AEGIS_ALLOW_NETWORK", "0") != "1":
                return {
                    "ok": False, "status": "failure", "tool": "execute_command",
                    "error": "capability_denied", "message": "Network capability is disabled for this sandbox.",
                    "capability": capability.value, "exit_code": None, "stdout": "", "stderr": "",
                    "duration_ms": 0, "truncated": False,
                    "sandbox": self.sandbox_status(),
                }
            if not approved:
                return {
                    "ok": False, "status": "failure", "tool": "execute_command",
                    "error": "approval_denied", "message": "Container command execution requires approval.",
                    "exit_code": None, "stdout": "", "stderr": "", "duration_ms": 0,
                    "truncated": False, "sandbox": self.sandbox_status(),
                }
            sandbox = self.container_sandbox or self.bwrap_sandbox
            sandbox_status = sandbox.status()
            if not sandbox_status.get("ready"):
                return {
                    "ok": False, "status": "failure", "tool": "execute_command",
                    "error": "sandbox_unavailable", "message": sandbox_status.get("message", "Sandbox unavailable."),
                    "exit_code": None, "stdout": "", "stderr": "", "duration_ms": 0,
                    "truncated": False, "sandbox": sandbox_status,
                }
            result = sandbox.run(command, target_cwd, timeout=timeout)
            result["approval"] = "approved"
            result["capability"] = capability.value
            return result

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
        safe_timeout = max(1, min(120, int(timeout)))
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
        timed_out = False
        try:
            native_result = native_process.run(tokens, cwd=target_cwd, env=env, timeout=safe_timeout)
            if native_result is not None:
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                stdout = native_result["stdout"]
                stderr = native_result["stderr"]
                exit_code = native_result["returncode"]
                timed_out = bool(native_result.get("timed_out"))
            # Production commands run in a fresh process group so a timeout
            # can terminate the complete child tree. The subprocess.run branch
            # keeps existing dependency-injection tests compatible.
            elif getattr(subprocess.run, "__module__", "subprocess") != "subprocess":
                proc = subprocess.run(tokens, cwd=str(target_cwd), env=env,
                                      capture_output=True, text=True,
                                      timeout=safe_timeout, check=False)
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                stdout, stderr = proc.stdout or "", proc.stderr or ""
                exit_code = proc.returncode
            else:
                proc = subprocess.Popen(tokens, cwd=str(target_cwd), env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        stdin=subprocess.DEVNULL, text=True,
                                        start_new_session=True)
                output_chunks: dict[str, list[str]] = {"stdout": [], "stderr": []}

                def _drain(stream: Any, channel: str) -> None:
                    if stream is None:
                        return
                    for line in iter(stream.readline, ""):
                        output_chunks[channel].append(line)
                        callback = self.command_event_callback
                        if callback:
                            callback({"event": "command_output", "stream": channel,
                                      "text": line[:200], "status": "running"})

                stdout_thread = threading.Thread(target=_drain, args=(proc.stdout, "stdout"), daemon=True)
                stderr_thread = threading.Thread(target=_drain, args=(proc.stderr, "stderr"), daemon=True)
                stdout_thread.start(); stderr_thread.start()
                try:
                    proc.wait(timeout=safe_timeout)
                    stdout_thread.join(timeout=2); stderr_thread.join(timeout=2)
                    stdout = "".join(output_chunks["stdout"])
                    stderr = "".join(output_chunks["stderr"])
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    stdout_thread.join(timeout=2); stderr_thread.join(timeout=2)
                    stdout = "".join(output_chunks["stdout"])
                    stderr = "".join(output_chunks["stderr"])
                    raw_out = getattr(exc, "output", None) or stdout or ""
                    raw_err = getattr(exc, "stderr", None) or stderr or ""
                    out = raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, bytes) else str(raw_out)
                    err = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, bytes) else str(raw_err)
                    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                    return {
                        "ok": False, "status": "failure", "tool": "execute_command",
                        "error": "command_timeout", "message": f"Command timed out after {safe_timeout} seconds",
                        "exit_code": proc.returncode, "stdout": out[:12000], "stderr": err[:12000],
                        "duration_ms": duration_ms, "truncated": len(out) > 12000 or len(err) > 12000,
                        "timed_out": True, "process_group_terminated": True,
                        "sandbox": sandbox,
                    }
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                exit_code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            raw_out = getattr(exc, "output", None) or getattr(exc, "stdout", "")
            raw_err = getattr(exc, "stderr", "")
            out = raw_out.decode("utf-8", errors="replace") if isinstance(raw_out, bytes) else str(raw_out or "")
            err = raw_err.decode("utf-8", errors="replace") if isinstance(raw_err, bytes) else str(raw_err or "")
            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            return {
                "ok": False,
                "status": "failure",
                "tool": "execute_command",
                "error": "command_timeout",
                "message": f"Command timed out after {safe_timeout} seconds",
                "exit_code": None,
                "stdout": (out or "")[:12000],
                "stderr": (err or "")[:12000],
                "duration_ms": duration_ms,
                "truncated": False,
                "timed_out": True,
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
            "timed_out": timed_out,
            "approval": "not_required" if not decision.requires_approval else "approved",
            "sandbox": {**sandbox, "status": "completed", "cwd": str(target_cwd),
                        "process_group": "isolated"},
        }
        if decision.requires_approval:
            res["approval"] = "approved"
        res.update(self._record_execution(command, target_cwd, res))
        return res

    def _record_execution(self, command: str, cwd: Path, result: dict[str, Any]) -> dict[str, Any]:
        """Persist bounded, non-secret execution evidence inside executions/."""
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        try:
            relative_cwd = str(cwd.relative_to(self.root)) or "."
        except ValueError:
            relative_cwd = "."
        record = {
            "execution_id": execution_id,
            "command": command,
            "cwd": relative_cwd,
            "status": result.get("status", "unknown"),
            "exit_code": result.get("exit_code"),
            "duration_ms": result.get("duration_ms", 0),
            "timed_out": bool(result.get("timed_out", False)),
            "stdout": str(result.get("stdout", ""))[-4000:],
            "stderr": str(result.get("stderr", ""))[-4000:],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.execution_dir.mkdir(parents=True, exist_ok=True)
        target = self.execution_dir / f"{execution_id}.json"
        try:
            target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        except OSError:
            return {"execution_id": execution_id, "execution_record": None}
        return {"execution_id": execution_id, "execution_record": str(target.relative_to(self.root))}

    def sandbox_status(self) -> dict[str, Any]:
        """Check the local command sandbox before any user command runs.

        AEGIS uses a constrained local subprocess backend: workspace-rooted
        cwd, sanitized environment, explicit command policy, fresh process
        group, output limits, and timeout cleanup. This preflight proves the
        interpreter can start and return a marker without touching user files.
        """
        try:
            if self.container_sandbox is not None:
                return self.container_sandbox.status()
            if self.bwrap_sandbox is not None:
                return self.bwrap_sandbox.status()
            root = self.root.resolve()
            if not root.exists() or not root.is_dir():
                return {"ready": False, "status": "unavailable", "backend": "local-process-group",
                        "message": "Workspace root is unavailable."}
            # Preserve deterministic dependency-injection tests that replace
            # subprocess.run with a controlled fake for command outcomes.
            if getattr(subprocess.run, "__module__", "subprocess") != "subprocess":
                return {"ready": True, "status": "ready", "backend": "local-process-group",
                        "workspace": str(root), "probe": "injected-test-backend",
                        "message": "Sandbox checks delegated to injected command backend."}
            probe = subprocess.run(
                [sys.executable, "-c", "print('AEGIS_SANDBOX_READY')"],
                cwd=str(root), capture_output=True, text=True, timeout=3,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONUNBUFFERED": "1"},
                check=False,
            )
            ready = probe.returncode == 0 and "AEGIS_SANDBOX_READY" in (probe.stdout or "")
            return {"ready": ready, "status": "ready" if ready else "unavailable",
                    "backend": "local-process-group", "workspace": str(root),
                    "probe_exit_code": probe.returncode,
                    "message": "Sandbox backend is ready." if ready else (probe.stderr or "probe failed")[:200]}
        except Exception as exc:
            return {"ready": False, "status": "unavailable", "backend": "local-process-group",
                    "message": f"Sandbox preflight failed: {type(exc).__name__}: {exc}"}

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
        def tree(path: str = ".", max_depth: int = 8, max_entries: int = 1000,
                 include_hidden: bool = False, include_generated: bool = False) -> dict[str, Any]:
            """Recursively show the bounded workspace file structure without reading file contents."""
            return self.tree(path, max_depth, max_entries, include_hidden, include_generated)
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
        def repository_context() -> dict[str, Any]:
            """Gather bounded repository structure, key contract files, and Git status."""
            return self.repository_context()
        @tool
        def git_status() -> dict[str, Any]:
            """Return read-only git status for the controlled workspace."""
            return self.git_status()
        @tool
        def git_diff() -> dict[str, Any]:
            """Return read-only git diff summary for the controlled workspace."""
            return self.git_diff()
        @tool
        def list_skills() -> dict[str, Any]:
            """List local bounded SKILL.md instructions available to agents."""
            return self.list_skills()
        @tool
        def read_skill(name: str) -> dict[str, Any]:
            """Read one bounded local skill instruction file."""
            return self.read_skill(name)
        @tool
        def workspace_diff(checkpoint_id: str) -> dict[str, Any]:
            """Show file-level and bounded unified changes since a checkpoint."""
            return self.workspace_diff(checkpoint_id)
        @tool
        def create_checkpoint(label: str = "checkpoint") -> dict[str, Any]:
            """Create an approved review checkpoint of the workspace."""
            return self.create_checkpoint(label)
        @tool
        def list_checkpoints() -> dict[str, Any]:
            """List available workspace review checkpoints."""
            return self.list_checkpoints()
        @tool
        def restore_checkpoint(checkpoint_id: str) -> dict[str, Any]:
            """Restore a checkpoint after explicit approval; this changes files."""
            return self.restore_checkpoint(checkpoint_id)
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
        def execute_command(command: str, cwd: str = ".", timeout: int = 90) -> dict[str, Any]:
            """Execute an allowed command (e.g. pytest) inside workspace root."""
            return self.execute_command(command, cwd=cwd, timeout=timeout)

        return [list_directory, tree, read_file, search_files, find_files, get_file_info, repository_context, git_status, git_diff, list_skills, read_skill, workspace_diff, create_checkpoint, list_checkpoints, restore_checkpoint, edit_file, create_file, create_python_script, execute_command]

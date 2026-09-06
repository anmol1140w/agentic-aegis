"""File operations tool — read, write, list within workspace boundaries.

All paths are validated through ``security.permissions`` before any
I/O.  Path traversal, symlink escape, and system directory access are
blocked.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from security.permissions import PermissionError_, validate_path

# The workspace root is resolved at import time from settings or env.
# Callers can override via ``set_workspace_root()``.
_workspace_root: Path = Path(
    os.environ.get("SIH_WORKSPACE", "./workspace")
).resolve()
_workspace_root.mkdir(parents=True, exist_ok=True)


def set_workspace_root(path: str | Path) -> None:
    """Override the workspace root (called during app initialisation)."""
    global _workspace_root
    _workspace_root = Path(path).resolve()
    _workspace_root.mkdir(parents=True, exist_ok=True)


def get_workspace_root() -> Path:
    return _workspace_root


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------

@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file inside the workspace.

    Args:
        file_path: Relative path within the workspace (e.g. "reports/q1.txt").
    """
    try:
        resolved = validate_path(file_path, _workspace_root, must_exist=True)
        if resolved.is_dir():
            return f"Error: '{file_path}' is a directory, not a file."
        size = resolved.stat().st_size
        if size > 10 * 1024 * 1024:  # 10 MB guard
            return f"Error: file is too large ({size:,} bytes). Max 10 MB."
        return resolved.read_text(encoding="utf-8", errors="replace")
    except PermissionError_ as exc:
        return f"Permission denied: {exc}"
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file inside the workspace. Creates parent dirs.

    Args:
        file_path: Relative path within the workspace.
        content: Text content to write.
    """
    try:
        resolved = validate_path(
            file_path, _workspace_root, allow_write=True,
        )
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {file_path}"
    except PermissionError_ as exc:
        return f"Permission denied: {exc}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool
def list_files(directory: str = ".") -> str:
    """List files and subdirectories inside a workspace directory.

    Args:
        directory: Relative path within the workspace. Defaults to root.
    """
    try:
        resolved = validate_path(
            directory, _workspace_root, must_exist=True,
        )
        if not resolved.is_dir():
            return f"Error: '{directory}' is not a directory."

        entries: list[str] = []
        for item in sorted(resolved.iterdir()):
            rel = item.relative_to(_workspace_root)
            kind = "DIR " if item.is_dir() else "FILE"
            size = ""
            if item.is_file():
                size = f"  ({item.stat().st_size:,} bytes)"
            entries.append(f"  {kind}  {rel}{size}")

        if not entries:
            return f"Directory '{directory}' is empty."
        header = f"Contents of {directory}/  ({len(entries)} items):\n"
        return header + "\n".join(entries)
    except PermissionError_ as exc:
        return f"Permission denied: {exc}"
    except Exception as exc:
        return f"Error listing directory: {exc}"


@tool
def create_directory(directory: str) -> str:
    """Create a directory (and parents) inside the workspace.

    Args:
        directory: Relative path within the workspace.
    """
    try:
        resolved = validate_path(directory, _workspace_root)
        resolved.mkdir(parents=True, exist_ok=True)
        return f"Created directory: {directory}"
    except PermissionError_ as exc:
        return f"Permission denied: {exc}"
    except Exception as exc:
        return f"Error creating directory: {exc}"

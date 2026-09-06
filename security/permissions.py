"""Path and permission validation for file operations.

Enforces workspace boundaries so generated code / tool calls cannot
escape into system directories or access secrets.
"""

from __future__ import annotations

import os
from pathlib import Path


# Patterns that must never appear in a resolved path
_BLOCKED_SUBSTRINGS: list[str] = [
    "/etc/",
    "/root/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/var/log/",
    "/.ssh/",
    "/.gnupg/",
    "/.aws/",
    "/.config/",
]

# Extensions that tools should never write
_DANGEROUS_EXTENSIONS: set[str] = {
    ".sh", ".bash", ".zsh", ".bat", ".cmd", ".ps1",
    ".exe", ".dll", ".so", ".dylib",
}


class PermissionError_(Exception):
    """Raised when a path or operation violates security policy."""


def validate_path(
    target: str | Path,
    workspace_root: str | Path,
    *,
    must_exist: bool = False,
    allow_write: bool = False,
) -> Path:
    """Validate that *target* is within *workspace_root* and safe.

    Parameters
    ----------
    target:
        The path to validate (may be relative or absolute).
    workspace_root:
        The trusted workspace root directory.
    must_exist:
        If True, raise if the resolved path does not exist.
    allow_write:
        If True, also check the file extension is not dangerous.

    Returns
    -------
    Path
        The resolved, validated absolute path.

    Raises
    ------
    PermissionError_
        If validation fails.
    """
    workspace = Path(workspace_root).resolve()
    resolved = (workspace / target).resolve()

    # 1. Must be under workspace
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise PermissionError_(
            f"Path escapes workspace boundary: {target!s}"
        )

    # 2. No blocked substrings
    resolved_str = str(resolved)
    for blocked in _BLOCKED_SUBSTRINGS:
        if blocked in resolved_str:
            raise PermissionError_(
                f"Path contains blocked segment '{blocked}': {target!s}"
            )

    # 3. No symlink escape (resolve already follows symlinks, check again)
    if resolved.is_symlink():
        real = resolved.resolve()
        try:
            real.relative_to(workspace)
        except ValueError:
            raise PermissionError_(
                f"Symlink escapes workspace: {target!s} → {real!s}"
            )

    # 4. Must exist check
    if must_exist and not resolved.exists():
        raise PermissionError_(f"Path does not exist: {resolved!s}")

    # 5. Dangerous extension check for writes
    if allow_write and resolved.suffix.lower() in _DANGEROUS_EXTENSIONS:
        raise PermissionError_(
            f"Writing files with extension '{resolved.suffix}' is not allowed"
        )

    return resolved


def is_path_safe(target: str | Path, workspace_root: str | Path) -> bool:
    """Non‑raising version of ``validate_path``."""
    try:
        validate_path(target, workspace_root)
        return True
    except PermissionError_:
        return False

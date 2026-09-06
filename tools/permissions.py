"""Session-scoped, canonical local filesystem permissions."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from security.permissions import PermissionError_, validate_path


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"


class SessionPermissions:
    """Keeps only in-memory directory grants for one workbench session."""

    def __init__(self, trusted_roots: list[str | Path] | None = None) -> None:
        self._grants: dict[AccessMode, set[Path]] = {AccessMode.READ: set(), AccessMode.WRITE: set()}
        for root in trusted_roots or []:
            resolved = Path(root).resolve()
            self._grants[AccessMode.READ].add(resolved)
            self._grants[AccessMode.WRITE].add(resolved)

    @staticmethod
    def canonical(path: str | Path) -> Path:
        """Resolve ``..`` and symlinks before every containment comparison."""
        return Path(path).expanduser().resolve(strict=False)

    def grant(self, path: str | Path, mode: AccessMode) -> Path:
        root = self.canonical(path)
        # Reuse the existing sensitive-path policy.  A grant must never weaken it.
        try:
            validate_path(root, root.parent if root.parent != root else root)
        except PermissionError_ as exc:
            raise PermissionError_(f"Path cannot be granted: {exc}") from exc
        self._grants[mode].add(root)
        if mode is AccessMode.WRITE:
            self._grants[AccessMode.READ].add(root)
        return root

    def allowed(self, path: str | Path, mode: AccessMode) -> bool:
        target = self.canonical(path)
        return any(self._contains(root, target) for root in self._grants[mode])

    def roots(self, mode: AccessMode) -> list[str]:
        return sorted(str(root) for root in self._grants[mode])

    @staticmethod
    def _contains(root: Path, target: Path) -> bool:
        try:
            target.relative_to(root)
            return True
        except ValueError:
            return False

"""Shared contracts and workspace confinement for office artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ArtifactError(ValueError):
    """A safe, user-facing artifact operation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ArtifactResult:
    status: str
    artifact_type: str
    path: str = ""
    size_bytes: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_workspace_path(raw: str | Path, workspace_root: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a path and reject absolute, traversal, and symlink escapes."""
    root = Path(workspace_root).resolve()
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactError("invalid_path", "Artifact path is outside the approved workspace") from exc
    if must_exist and (not resolved.is_file()):
        raise ArtifactError("invalid_input", f"Artifact input does not exist: {raw}")
    return resolved


def output_path(raw: str | Path | None, workspace_root: str | Path, *, artifact_type: str, stem: str) -> Path:
    root = Path(workspace_root).resolve()
    if raw is None or not str(raw).strip():
        path = root / "outputs" / "artifacts" / f"{stem}.{artifact_type}"
    else:
        path = resolve_workspace_path(raw, root)
        if path.suffix.lower() != f".{artifact_type}":
            path = path.with_suffix(f".{artifact_type}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_spec(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ArtifactError("invalid_specification", "Artifact specification must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactError("invalid_specification", "Artifact specification must be an object")
    return value

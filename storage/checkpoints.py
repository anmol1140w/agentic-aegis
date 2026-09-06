"""Reviewable workspace checkpoints inspired by editor agent workflows.

Checkpoints live inside the controlled workspace, exclude repository metadata
and the checkpoint store itself, and never execute hooks. Restore is explicit
and is expected to be approval-gated by the caller.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkspaceCheckpointStore:
    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.store = self.workspace / ".aegis" / "checkpoints"

    def create(self, label: str = "checkpoint") -> dict[str, Any]:
        checkpoint_id = f"cp_{uuid.uuid4().hex[:12]}"
        self.store.mkdir(parents=True, exist_ok=True)
        archive = self.store / f"{checkpoint_id}.tar.gz"
        manifest = self._manifest()
        with tarfile.open(archive, "w:gz") as tar:
            for rel in manifest:
                tar.add(self.workspace / rel, arcname=rel, recursive=False)
        metadata = {
            "checkpoint_id": checkpoint_id,
            "label": self._safe_label(label),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(manifest),
            "manifest": manifest,
            "archive": archive.name,
        }
        (self.store / f"{checkpoint_id}.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return metadata

    def list(self) -> list[dict[str, Any]]:
        if not self.store.is_dir():
            return []
        records = []
        for path in sorted(self.store.glob("cp_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def get(self, checkpoint_id: str) -> dict[str, Any]:
        if not checkpoint_id.startswith("cp_") or "/" in checkpoint_id or "\\" in checkpoint_id:
            raise ValueError("invalid checkpoint id")
        path = self.store / f"{checkpoint_id}.json"
        if not path.is_file():
            raise FileNotFoundError(checkpoint_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def diff(self, checkpoint_id: str) -> dict[str, Any]:
        record = self.get(checkpoint_id)
        before = record.get("manifest", {})
        after = self._manifest()
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        modified = sorted(path for path in set(before) & set(after) if before[path]["sha256"] != after[path]["sha256"])
        patches: dict[str, str] = {}
        for rel in modified:
            old = self._read_archive_text(record, rel)
            new = (self.workspace / rel).read_text(encoding="utf-8", errors="replace")
            if old is not None:
                patches[rel] = "".join(difflib.unified_diff(
                    old.splitlines(keepends=True), new.splitlines(keepends=True),
                    fromfile=f"checkpoint/{rel}", tofile=f"workspace/{rel}",
                ))[:64 * 1024]
        return {"checkpoint_id": checkpoint_id, "added": added, "removed": removed,
                "modified": modified, "patches": patches,
                "changed_files": sorted(set(added + removed + modified))}

    def restore(self, checkpoint_id: str) -> dict[str, Any]:
        record = self.get(checkpoint_id)
        archive = self.store / record["archive"]
        if not archive.is_file():
            raise FileNotFoundError(str(archive))
        with tempfile.TemporaryDirectory(prefix="aegis-restore-") as temp:
            temp_root = Path(temp)
            with tarfile.open(archive, "r:gz") as tar:
                members = tar.getmembers()
                for member in members:
                    target = (temp_root / member.name).resolve()
                    if not target.is_relative_to(temp_root) or member.issym() or member.islnk() or member.isdev():
                        raise ValueError("unsafe checkpoint archive")
                if sys.version_info >= (3, 12):
                    tar.extractall(temp_root, filter="data")
                else:
                    tar.extractall(temp_root)
            before = self._manifest()
            target_files = set(record.get("manifest", {}))
            for rel in set(before) - target_files:
                (self.workspace / rel).unlink(missing_ok=True)
            for rel in target_files:
                source = temp_root / rel
                destination = self.workspace / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        return {"checkpoint_id": checkpoint_id, "restored_files": len(record.get("manifest", {}))}

    def _manifest(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for path in self.workspace.rglob("*"):
            if not path.is_file() or ".git" in path.parts or ".aegis" in path.parts or path.is_symlink():
                continue
            rel = str(path.relative_to(self.workspace))
            data = path.read_bytes()
            result[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        return dict(sorted(result.items()))

    def _read_archive_text(self, record: dict[str, Any], rel: str) -> str | None:
        try:
            with tarfile.open(self.store / record["archive"], "r:gz") as tar:
                member = tar.getmember(rel)
                if member.size > 256 * 1024:
                    return None
                handle = tar.extractfile(member)
                return handle.read().decode("utf-8", errors="replace") if handle else None
        except (KeyError, OSError, tarfile.TarError):
            return None

    @staticmethod
    def _safe_label(label: str) -> str:
        cleaned = " ".join(str(label).split())[:120]
        return cleaned or "checkpoint"

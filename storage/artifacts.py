"""Small provenance-aware artifact catalog for local, verified outputs."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ArtifactManager:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "artifacts.json"

    def register_artifact(self, path: str | Path, *, artifact_type: str = "file",
                          format: str = "", execution_id: str = "", agent_id: str = "",
                          tool_id: str = "", verification_status: str = "verified",
                          metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        target = Path(path).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("artifact path is outside the artifact root")
        if not target.is_file():
            raise FileNotFoundError(str(target))
        record = {"artifact_id": f"artifact_{uuid.uuid4().hex[:12]}", "artifact_type": artifact_type,
                  "path": str(target), "format": format or target.suffix.lstrip("."),
                  "size": target.stat().st_size, "created_at": datetime.now(timezone.utc).isoformat(),
                  "producer_agent": agent_id, "producer_tool": tool_id,
                  "execution_id": execution_id, "verification_status": verification_status,
                  "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "metadata": metadata or {}}
        records = json.loads(self._index.read_text()) if self._index.exists() else []
        records.append(record)
        self._index.write_text(json.dumps(records, indent=2) + "\n")
        return record

    def list_artifacts(self) -> list[dict[str, Any]]:
        return json.loads(self._index.read_text()) if self._index.exists() else []

    def find_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        return next((a for a in self.list_artifacts() if a["artifact_id"] == artifact_id), None)

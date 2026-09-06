"""Persistent per-run output and trace storage."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime.errors import OutputPersistenceError


class OutputStore:
    """Write visible results and non-sensitive execution metadata per run."""

    def __init__(self, workspace_dir: str | Path | None = None):
        root = Path(workspace_dir or Path(__file__).resolve().parents[1] / "workspace")
        self.workspace_dir = root.resolve()
        self.outputs_dir = self.workspace_dir / "outputs"
        self.artifacts_dir = self.workspace_dir / "artifacts"
        for directory in (self.workspace_dir / "inputs", self.outputs_dir,
                          self.artifacts_dir, self.workspace_dir / "executions",
                          self.workspace_dir / "temporary"):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_run_id() -> str:
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

    def save_run(
        self,
        *,
        run_id: str,
        result: str,
        metadata: dict[str, Any],
        trace: list[dict[str, Any]],
    ) -> Path:
        """Persist the mandatory result, metadata, and trace files atomically enough for CLI use."""
        run_dir = self.outputs_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "result.txt").write_text(result, encoding="utf-8")
            (run_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (run_dir / "trace.json").write_text(
                json.dumps({"run_id": run_id, "events": trace}, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise OutputPersistenceError(f"Could not save run output: {exc}") from exc
        return run_dir

"""Policy-facing artifact service used by AEGIS specialists."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable

from .common import ArtifactError, ArtifactResult, json_spec, resolve_workspace_path
from .presentation import create_presentation, edit_presentation, validate_presentation
from .spreadsheet import create_workbook, edit_workbook, validate_workbook


class ArtifactService:
    """Deterministic office artifact operations with no model-generated code."""

    def __init__(self, workspace_root: str | Path, *, audit: Any | None = None,
                 progress_callback: Callable[[dict[str, Any]], None] | None = None):
        self.workspace_root = Path(workspace_root).resolve()
        self.audit = audit
        self.progress_callback = progress_callback

    def _progress(self, event: str, **data: Any) -> None:
        if self.progress_callback:
            self.progress_callback({"event": event, **data})

    def _audit(self, operation: str, artifact_type: str, result: dict[str, Any], started: float, **metadata: Any) -> None:
        if self.audit:
            self.audit.log("artifact_operation", action=operation, tool=f"artifact.{artifact_type}", status=result.get("status"), duration_ms=round((time.monotonic() - started) * 1000, 2), metadata={"artifact_type": artifact_type, "path": result.get("path", ""), "validation": result.get("validation", {}), **metadata})

    def create_presentation(self, spec: dict[str, Any] | str, output_path: str | None = None) -> dict[str, Any]:
        started = time.monotonic(); self._progress("artifact_specification_created", artifact_type="pptx")
        try:
            result = create_presentation(spec, output_path, self.workspace_root, progress=lambda event, **data: self._progress(event, **data))
        except ArtifactError as exc:
            result = ArtifactResult("failure", "pptx", errors=[f"{exc.code}:{exc.message}"]).to_dict()
        except Exception as exc:
            result = ArtifactResult("failure", "pptx", errors=[f"artifact_generation_failed:{type(exc).__name__}"]).to_dict()
        self._audit("create", "pptx", result, started); return result

    def create_workbook(self, spec: dict[str, Any] | str, output_path: str | None = None) -> dict[str, Any]:
        started = time.monotonic(); self._progress("artifact_specification_created", artifact_type="xlsx")
        try:
            result = create_workbook(spec, output_path, self.workspace_root, progress=lambda event, **data: self._progress(event, **data))
        except ArtifactError as exc:
            result = ArtifactResult("failure", "xlsx", errors=[f"{exc.code}:{exc.message}"]).to_dict()
        except Exception as exc:
            result = ArtifactResult("failure", "xlsx", errors=[f"artifact_generation_failed:{type(exc).__name__}"]).to_dict()
        self._audit("create", "xlsx", result, started); return result

    def edit(self, source_path: str, operations: list[dict[str, Any]], output_path: str | None = None) -> dict[str, Any]:
        started = time.monotonic(); source = resolve_workspace_path(source_path, self.workspace_root, must_exist=True)
        artifact_type = source.suffix.lower().lstrip(".")
        try:
            if artifact_type == "pptx": result = edit_presentation(source, operations, output_path, self.workspace_root)
            elif artifact_type == "xlsx": result = edit_workbook(source, operations, output_path, self.workspace_root)
            else: raise ArtifactError("invalid_input", "Only PPTX and XLSX editing is supported")
        except ArtifactError as exc:
            result = ArtifactResult("failure", artifact_type, errors=[f"{exc.code}:{exc.message}"]).to_dict()
        self._audit("edit", artifact_type, result, started, source=str(source)); return result

    def validate(self, path: str, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.monotonic(); target = resolve_workspace_path(path, self.workspace_root, must_exist=True); artifact_type = target.suffix.lower().lstrip(".")
        self._progress("artifact_validation_started", artifact_type=artifact_type, path=str(target))
        if artifact_type == "pptx": result = validate_presentation(target, workspace_root=self.workspace_root, expected=expected)
        elif artifact_type == "xlsx": result = validate_workbook(target, workspace_root=self.workspace_root, expected=expected)
        else: raise ArtifactError("invalid_input", "Unsupported artifact type")
        self._progress("artifact_validation_passed" if result.get("status") == "passed" else "artifact_validation_failed", artifact_type=artifact_type, validation=result)
        self._audit("validate", artifact_type, result, started); return result

    def generate_from_request(self, request: str, artifact_type: str) -> dict[str, Any]:
        """Create a conservative spec from intent; no Python or binary code comes from the model."""
        text = request.strip(); lower = text.lower(); safe_title = re.sub(r"\s+", " ", text).strip(" .")[:90] or "AEGIS Artifact"
        if artifact_type == "pptx":
            count_match = re.search(r"\b(\d+)\s*[- ]?slide", lower); count = max(1, min(int(count_match.group(1)), 30)) if count_match else 5
            slides = [{"layout": "title", "title": safe_title, "subtitle": "Generated locally by AEGIS"}]
            slides += [{"layout": "content", "title": f"AEGIS overview {index}", "bullets": ["Local-first execution", "Capability-based routing", "Deterministic validation"]} for index in range(2, count + 1)]
            return self.create_presentation({"title": safe_title, "subtitle": "Sovereign AI Workbench", "theme": "professional", "slides": slides})
        rows = [["January", "Operations", 0], ["February", "Operations", 0], ["March", "Operations", 0], ["April", "Operations", 0]]
        return self.create_workbook({"title": safe_title, "worksheets": [{"name": "Summary", "headers": ["Month", "Category", "Amount"], "rows": rows, "table": {"name": "ExpenseTable"}, "charts": [{"type": "column", "title": "Monthly expenses", "data_range": "A1:C5", "anchor": "E2"}], "freeze_panes": "A2"}]})

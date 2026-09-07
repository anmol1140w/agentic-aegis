"""Authoritative policy-aware workspace tool runtime.

Legacy method names remain available for existing agents. New integrations
should call ``ToolRuntime.invoke`` with a namespaced operation.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from tools.contracts import (CheckpointInput, CommandInput, CreateInput, EditInput,
                             LooseInput, ReadInput, RestoreInput, SearchInput, ToolResult)


class ToolRuntime:
    OPERATIONS: dict[str, tuple[str, type[BaseModel], str, bool, str]] = {
        "workspace.read.list": ("list_directory", LooseInput, "read_only", False, "list workspace entries"),
        "workspace.read.tree": ("tree", LooseInput, "read_only", False, "show bounded workspace tree"),
        "workspace.read.read": ("read_file", ReadInput, "read_only", False, "read one workspace file"),
        "workspace.read.find": ("find_files", LooseInput, "read_only", False, "find workspace files"),
        "workspace.read.search": ("search_files", SearchInput, "read_only", False, "search workspace text"),
        "workspace.read.stat": ("get_file_info", ReadInput, "read_only", False, "stat a workspace path"),
        "workspace.read.context": ("repository_context", LooseInput, "read_only", False, "collect bounded repository facts"),
        "workspace.verify.status": ("git_status", LooseInput, "read_only", False, "read Git status"),
        "workspace.verify.diff": ("git_diff", LooseInput, "read_only", False, "read Git diff summary"),
        "workspace.change.edit": ("edit_file", EditInput, "reversible", True, "replace one exact text span"),
        "workspace.change.create": ("create_file", CreateInput, "reversible", True, "create one workspace file"),
        "workspace.run.command": ("execute_command", CommandInput, "external_side_effect", True, "run a policy-checked command"),
        "workspace.checkpoint.create": ("create_checkpoint", CheckpointInput, "reversible", True, "create a checkpoint"),
        "workspace.checkpoint.list": ("list_checkpoints", LooseInput, "read_only", False, "list checkpoints"),
        "workspace.checkpoint.restore": ("restore_checkpoint", RestoreInput, "destructive", True, "restore a checkpoint"),
    }

    def __init__(self, workspace: Any):
        self.workspace = workspace

    @classmethod
    def catalog(cls) -> list[dict[str, Any]]:
        return [{"name": name, "implementation": impl, "mutation": mutation,
                 "requires_approval": approval, "description": description,
                 "workspace_scoped": True, "supports_dry_run": name in {"workspace.change.edit", "workspace.change.create"}}
                for name, (impl, _, mutation, approval, description) in cls.OPERATIONS.items()]

    def invoke(self, name: str, arguments: dict[str, Any] | None = None, *, request_id: str | None = None) -> dict[str, Any]:
        request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        spec = self.OPERATIONS.get(name)
        if spec is None:
            return ToolResult(ok=False, tool=name, operation="unknown", request_id=request_id,
                              error={"code": "invalid_tool", "message": f"Unknown tool: {name}", "remediation": "Use runtime.catalog()."}).model_dump()
        implementation, schema, mutation, _, _ = spec
        raw = arguments or {}
        try:
            # Empty schemas use BaseModel only as a marker; tolerate the
            # legacy default-argument calls for read-only operations.
            parsed = schema.model_validate(raw)
        except ValidationError as exc:
            return ToolResult(ok=False, tool=name, operation=implementation, request_id=request_id,
                              error={"code": "invalid_input", "message": str(exc)[:1000], "remediation": "Correct arguments to match the tool schema."}).model_dump()
        args = parsed.model_dump()
        dry_run = bool(args.pop("dry_run", False))
        if implementation == "edit_file":
            args = {"path": args.pop("path"), "old_text": args.pop("find"), "new_text": args.pop("replace"),
                    "expected_matches": parsed.expected_matches, "dry_run": dry_run}
        try:
            result = getattr(self.workspace, implementation)(**args)
        except TypeError:
            # Compatibility with injected test doubles that implement the
            # old three-argument method signatures.
            args.pop("expected_matches", None); args.pop("dry_run", None)
            result = getattr(self.workspace, implementation)(**args)
        return self._envelope(name, implementation, request_id, result)

    @staticmethod
    def _envelope(name: str, operation: str, request_id: str, result: Any) -> dict[str, Any]:
        result = result if isinstance(result, dict) else {"result": result}
        ok = bool(result.get("ok", False))
        payload = dict(result)
        payload.pop("ok", None); payload.pop("tool", None); payload.pop("status", None)
        if ok:
            return ToolResult(ok=True, tool=name, operation=operation, request_id=request_id,
                              data=payload, evidence={"operation": operation}).model_dump()
        code = str(result.get("error", "tool_failure"))
        retryable = code in {"timeout", "command_timeout", "tool_failure", "filesystem_error"}
        return ToolResult(ok=False, tool=name, operation=operation, request_id=request_id,
                          data=payload, error={"code": code, "message": str(result.get("message", code))[:1000],
                                               "retryable": retryable}).model_dump()

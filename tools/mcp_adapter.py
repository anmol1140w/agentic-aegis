"""Optional local MCP-shaped adapter over the authoritative AEGIS ToolRegistry.

This module deliberately does not start a server or add another executor. It
provides the same discover/call shape an MCP transport can expose later while
delegating every operation to the existing registry and policy-bound tool.
"""
from __future__ import annotations

from typing import Any

from runtime.permissions import PermissionManager


class AegisMCPAdapter:
    def __init__(self, registry: Any, *, permissions: PermissionManager | None = None, session_id: str = "mcp") -> None:
        self.registry = registry
        self.permissions = permissions
        self.session_id = session_id

    def list_tools(self, *, tags: set[str] | None = None) -> list[dict[str, Any]]:
        tools = []
        for item in self.registry.list_tools():
            meta = self.registry.get_meta(item.name)
            if tags and not (tags & set(meta.tags)):
                continue
            tools.append({"name": item.name, "description": item.description or "",
                          "category": meta.category, "tags": list(meta.tags),
                          "requires_approval": meta.requires_approval,
                          "permissions": dict(meta.permissions), "offline_capable": meta.offline_capable})
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return {"success": False, "tool": name, "error": "invalid_arguments"}
        try:
            tool = self.registry.get(name)
        except KeyError:
            return {"success": False, "tool": name, "error": "unknown_tool"}
        if self.permissions is not None:
            meta = self.registry.get_meta(name)
            resources = [str(arguments.get(key)) for key in ("path", "cwd", "checkpoint_id") if arguments.get(key)] or [name]
            request = self.permissions.request(self.session_id, name, resources, metadata={"requires_approval": meta.requires_approval})
            if request.status != "approved":
                return {"success": False, "tool": name, "error": "permission_required",
                        "request_id": request.request_id, "status": request.status}
        try:
            result = await tool.ainvoke(arguments)
            if isinstance(result, dict):
                return {"success": bool(result.get("ok", result.get("success", True))), "tool": name, **result}
            return {"success": True, "tool": name, "data": result}
        except Exception as exc:
            return {"success": False, "tool": name, "error": type(exc).__name__, "message": str(exc)[:300]}

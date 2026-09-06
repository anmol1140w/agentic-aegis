import pytest
from langchain_core.tools import tool

from runtime.permissions import PermissionEffect, PermissionManager, PermissionRule
from tools.mcp_adapter import AegisMCPAdapter
from tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_adapter_returns_permission_request_before_execution() -> None:
    @tool
    def write_fixture(path: str) -> dict:
        """Write a fixture."""
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(write_fixture, requires_approval=True, permissions={"read": False, "write": True})
    manager = PermissionManager()
    adapter = AegisMCPAdapter(registry, permissions=manager, session_id="s1")
    result = await adapter.call_tool("write_fixture", {"path": "workspace/a.txt"})
    assert result["error"] == "permission_required"
    request = manager.get(result["request_id"])
    assert request is not None
    manager.reply(request.request_id, True, save=True)
    allowed = await adapter.call_tool("write_fixture", {"path": "workspace/a.txt"})
    assert allowed["success"] is True

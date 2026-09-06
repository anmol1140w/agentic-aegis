import pytest
from langchain_core.tools import tool

from tools.mcp_adapter import AegisMCPAdapter
from tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_mcp_adapter_delegates_to_native_registry():
    registry = ToolRegistry()

    @tool
    def local_probe(value: str) -> dict:
        """Return a local structured probe result."""
        return {"ok": True, "value": value}

    registry.register(local_probe, tags=["local"])
    adapter = AegisMCPAdapter(registry)
    assert adapter.list_tools(tags={"local"})[0]["name"] == "local_probe"
    result = await adapter.call_tool("local_probe", {"value": "ok"})
    assert result["success"] is True
    assert result["value"] == "ok"
    assert (await adapter.call_tool("missing", {}))["error"] == "unknown_tool"

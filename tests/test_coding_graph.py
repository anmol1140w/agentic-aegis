import pytest
from langchain_core.tools import tool

from runtime.coding_graph import run_coding_graph


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def stream_chat(self, messages, **kwargs):
        yield next(self.responses)


class FakeRegistry:
    def __init__(self):
        @tool
        def read_file(path: str) -> dict:
            """Read a workspace file."""
            return {"ok": True, "tool": "read_file", "path": path, "content": "source"}
        self.tools = {"read_file": read_file}

    def list_names(self):
        return list(self.tools)

    def get(self, name):
        return self.tools[name]


@pytest.mark.asyncio
async def test_native_coding_graph_returns_tool_result_before_final():
    state = await run_coding_graph(
        FakeProvider([
            '{"action":"tool","tool":"read_file","arguments":{"path":"main.py"}}',
            '{"action":"final","answer":"verified"}',
        ]),
        FakeRegistry(),
        "Inspect main.py",
    )
    assert state["terminal_status"] == "success"
    assert len(state["tool_results"]) == 1
    assert state["tool_results"][0]["content"] == "source"


@pytest.mark.asyncio
async def test_native_coding_graph_bounds_invalid_actions():
    state = await run_coding_graph(
        FakeProvider(["not json", "still not json", "also not json"]),
        FakeRegistry(),
        "Inspect files",
        max_invalid_actions=3,
    )
    assert state["terminal_status"] == "blocked"
    assert state["invalid_actions"] == 3

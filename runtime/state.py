"""LangGraph agent state definition.

Uses serialization-safe primitive types (dicts, lists, strings, numbers, booleans)
to prevent msgpack/checkpoint deserialization warnings in LangGraph.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Trace entry helper
# ---------------------------------------------------------------------------

class TraceEntry(BaseModel):
    """Pydantic model for internal trace creation and validation."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    step: str                # e.g. "classify", "route", "tool", "execute"
    message: str
    duration_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_serializable_dict(self) -> dict[str, Any]:
        """Convert to pure primitive dict for LangGraph state."""
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


def make_trace_dict(
    step: str,
    message: str,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Helper to produce a serialization-safe trace dictionary."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "message": message,
        "duration_ms": duration_ms,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Custom reducers
# ---------------------------------------------------------------------------

def _append_traces(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Reducer: append new serialization-safe trace dicts to existing list."""
    return list(existing or []) + list(new or [])


def _append_strings(
    existing: list[str] | None,
    new: list[str] | None,
) -> list[str]:
    """Reducer: append strings."""
    return list(existing or []) + list(new or [])


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """State flowing through the LangGraph orchestrator.

    All persisted state is strictly composed of:
    - BaseMessage / AnyMessage (natively supported by LangGraph)
    - dict[str, Any]
    - list[...]
    - str, int, float, bool
    """

    # -- Core conversation --
    messages: Annotated[list[AnyMessage], add_messages]

    # -- Task metadata (stored as primitive dict) --
    task: dict[str, Any] | None

    # -- Routing --
    execution_mode: str             # "direct_tool", "model", "model_with_tools"
    selected_model: str             # registry name, e.g. "qwen-coder" or "direct_tool"
    selected_model_id: str          # Ollama tag, e.g. "qwen2.5-coder:7b" or "none"
    routing_reason: str
    is_direct_tool: bool

    # -- Execution tracking & Latency Breakdown (ms) --
    current_step: str
    iteration: int
    classification_ms: float
    routing_ms: float
    tool_ms: float
    model_ms: float
    image_load_ms: float
    model_first_token_ms: float
    total_ms: float
    run_id: str
    output_dir: str
    run_started_at: str

    # -- Accumulating lists --
    trace: Annotated[list[dict[str, Any]], _append_traces]
    errors: Annotated[list[str], _append_strings]

    # -- Files --
    attached_files: list[str]

    previous_task_type: str | None

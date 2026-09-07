"""Typed public contracts for the workspace tool runtime."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    remediation: str = ""


class ToolResult(BaseModel):
    """Stable envelope used by namespaced tools and MCP adapters."""
    model_config = ConfigDict(extra="allow")
    ok: bool
    tool: str
    operation: str
    request_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    error: ToolError | None = None


class ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


class LooseInput(BaseModel):
    model_config = ConfigDict(extra="allow")


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)
    path: str = "."


class EditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    find: str = Field(min_length=1)
    replace: str
    expected_matches: int = Field(default=1, ge=1, le=20)
    dry_run: bool = False


class CreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    content: str = Field(max_length=64 * 1024)
    dry_run: bool = False


class CommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    command: str = Field(min_length=1, max_length=4000)
    cwd: str = "."
    timeout: int = Field(default=90, ge=1, le=120)


class CheckpointInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(default="checkpoint", max_length=200)


class RestoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checkpoint_id: str = Field(min_length=1, max_length=200)

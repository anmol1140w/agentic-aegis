"""Tool registry — central catalog of all agent tools.

Every tool is a LangChain ``@tool``-decorated function registered here.
The orchestrator binds the tools relevant to the current task to the
chat model before invoking it.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Risk level of a tool — determines whether approval is needed."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolMeta(BaseModel):
    """Metadata about a registered tool."""
    name: str
    description: str
    risk_level: RiskLevel = RiskLevel.LOW
    requires_approval: bool = False
    timeout_seconds: int = 30
    tags: list[str] = Field(default_factory=list)
    category: str = "general"
    status: str = "implemented"
    permissions: dict[str, bool] = Field(default_factory=lambda: {
        "read": True, "write": False, "delete": False, "network": False,
        "external_side_effect": False, "credential_access": False, "system_access": False,
    })
    reversible: bool = True
    idempotent: bool = True
    offline_capable: bool = True


class ToolRegistry:
    """Central registry of available tools.

    Tools are stored as LangChain ``BaseTool`` instances alongside
    their metadata.  The orchestrator queries this registry to bind
    task-appropriate tools to the LLM.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._meta: dict[str, ToolMeta] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        risk_level: RiskLevel = RiskLevel.LOW,
        requires_approval: bool = False,
        timeout_seconds: int = 30,
        tags: list[str] | None = None,
        category: str = "general",
        permissions: dict[str, bool] | None = None,
        reversible: bool = True,
        idempotent: bool = True,
        offline_capable: bool = True,
    ) -> None:
        """Register a LangChain tool with metadata."""
        name = tool.name
        self._tools[name] = tool
        self._meta[name] = ToolMeta(
            name=name,
            description=tool.description or "",
            risk_level=risk_level,
            requires_approval=requires_approval,
            timeout_seconds=timeout_seconds,
            tags=tags or [],
            category=category,
            permissions=permissions or {"read": True, "write": False, "delete": False,
                                        "network": False, "external_side_effect": False,
                                        "credential_access": False, "system_access": False},
            reversible=reversible, idempotent=idempotent, offline_capable=offline_capable,
        )

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    def get_meta(self, name: str) -> ToolMeta:
        return self._meta[name]

    def list_tools(self) -> list[BaseTool]:
        """Return all registered LangChain tools."""
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tools_by_tags(self, *tags: str) -> list[BaseTool]:
        """Return tools that have at least one of the given tags."""
        tag_set = set(tags)
        return [
            self._tools[name]
            for name, meta in self._meta.items()
            if tag_set.intersection(meta.tags)
        ]

    def get_safe_tools(self) -> list[BaseTool]:
        """Return only LOW-risk tools (no approval needed)."""
        return [
            self._tools[name]
            for name, meta in self._meta.items()
            if meta.risk_level == RiskLevel.LOW
        ]

    def search(self, query: str) -> list[ToolMeta]:
        """Discover tools by bounded name/description/tag matching."""
        needle = query.lower().strip()
        return [m for m in self._meta.values()
                if needle in m.name.lower() or needle in m.description.lower()
                or any(needle in tag.lower() for tag in m.tags)]

    def filter_by_permission(self, permission: str, value: bool = True) -> list[ToolMeta]:
        return [m for m in self._meta.values() if m.permissions.get(permission, False) is value]

    def filter_by_risk(self, risk: RiskLevel) -> list[ToolMeta]:
        return [m for m in self._meta.values() if m.risk_level == risk]

    def check_available(self, name: str) -> bool:
        return name in self._tools and self._meta[name].status == "implemented"

"""Bounded parallel workforce execution over the existing AgentRegistry."""

from __future__ import annotations

import asyncio
from typing import Any

from runtime.agents import AgentRegistry, AgentRequest, AgentResult, AgentStatus


class WorkforceCoordinator:
    def __init__(self, registry: AgentRegistry, *, max_parallel: int = 3) -> None:
        self.registry = registry
        self.max_parallel = max(1, min(8, max_parallel))

    async def run(self, assignments: list[dict[str, Any]]) -> list[AgentResult]:
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def execute(item: dict[str, Any]) -> AgentResult:
            agent_name = str(item.get("agent", ""))
            try:
                agent = self.registry.get(agent_name)
            except KeyError:
                return AgentResult(agent=agent_name, status=AgentStatus.FAILURE, summary="Unknown workforce agent", errors=["agent_not_registered"])
            request = AgentRequest(task=str(item.get("task", "")), context=dict(item.get("context") or {}), constraints=list(item.get("constraints") or []))
            async with semaphore:
                return await agent.run(request)

        return list(await asyncio.gather(*(execute(item) for item in assignments)))

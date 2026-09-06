import asyncio

from runtime.agents import AgentDescriptor, AgentRegistry, AgentResult, AgentStatus, BaseAgent
from runtime.workforce import WorkforceCoordinator


class FakeAgent(BaseAgent):
    def __init__(self, name: str):
        self.descriptor = AgentDescriptor(name=name, role="test", capabilities=[], provider_name="local")

    async def run(self, request):
        return AgentResult(agent=self.descriptor.name, status=AgentStatus.SUCCESS, summary=request.task)


def test_workforce_runs_registered_agents_and_reports_unknown() -> None:
    registry = AgentRegistry()
    registry.register(FakeAgent("one"))
    result = asyncio.run(WorkforceCoordinator(registry).run([
        {"agent": "one", "task": "first"},
        {"agent": "missing", "task": "second"},
    ]))
    assert [item.status for item in result] == [AgentStatus.SUCCESS, AgentStatus.FAILURE]
    assert result[0].summary == "first"

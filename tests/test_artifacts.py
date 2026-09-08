from __future__ import annotations

from pathlib import Path

import pytest

from runtime.agents import MasterAgent
from runtime.compound_tasks import decompose_task
from tools.artifacts import ArtifactError, ArtifactService
from tools.artifacts.presentation import validate_presentation
from tools.artifacts.spreadsheet import validate_workbook


def test_create_and_validate_presentation(tmp_path: Path):
    service = ArtifactService(tmp_path)
    result = service.create_presentation({
        "title": "AEGIS", "theme": "technical",
        "slides": [{"layout": "title", "title": "AEGIS"}, {"title": "Architecture", "bullets": ["Master", "Tools"]}],
    }, "outputs/aegis.pptx")
    assert result["status"] == "success"
    assert result["validation"]["status"] == "passed"
    assert result["validation"]["slides"] == 2
    assert Path(result["path"]).is_relative_to(tmp_path)


def test_presentation_empty_spec_rejected(tmp_path: Path):
    with pytest.raises(ArtifactError):
        from tools.artifacts.presentation import create_presentation
        create_presentation({"slides": []}, "outputs/empty.pptx", tmp_path)


def test_presentation_edit_and_invalid_slide(tmp_path: Path):
    service = ArtifactService(tmp_path)
    created = service.create_presentation({"title": "Original", "slides": [{"layout": "title", "title": "Original"}]}, "outputs/source.pptx")
    edited = service.edit(created["path"], [{"operation": "replace_text", "slide": 0, "old_text": "Original", "new_text": "Updated"}], "outputs/edited.pptx")
    assert edited["status"] == "success"
    assert "Updated" in validate_presentation(edited["path"], workspace_root=tmp_path)["titles"][0]
    invalid = service.edit(created["path"], [{"operation": "update_text", "slide": 10, "text": "bad"}], "outputs/bad.pptx")
    assert invalid["status"] == "failure"


def test_create_workbook_with_formula_table_and_chart(tmp_path: Path):
    service = ArtifactService(tmp_path)
    result = service.create_workbook({
        "title": "Budget", "worksheets": [{
            "name": "Summary", "headers": ["Month", "Amount", "Total"],
            "rows": [["January", 100, None], ["February", 150, None]],
            "formulas": {"C2": "=B2", "C3": "=B3"},
            "table": {"name": "BudgetTable"},
            "charts": [{"type": "column", "title": "Budget", "data_range": "A1:B3"}],
        }],
    }, "outputs/budget.xlsx")
    assert result["status"] == "success"
    assert result["validation"]["status"] == "passed"
    assert result["validation"]["sheets"] == ["Summary"]
    assert result["validation"]["charts"] == 1
    assert result["validation"]["tables"] == 1


def test_workbook_edit_and_invalid_sheet(tmp_path: Path):
    service = ArtifactService(tmp_path)
    created = service.create_workbook({"worksheets": [{"name": "Data", "headers": ["A"], "rows": [[1]]}]}, "outputs/data.xlsx")
    edited = service.edit(created["path"], [{"operation": "write_cell", "sheet": "Data", "cell": "A3", "value": "=SUM(A2)"}], "outputs/edited.xlsx")
    assert edited["status"] == "success"
    assert validate_workbook(edited["path"], workspace_root=tmp_path)["status"] == "passed"
    invalid = service.edit(created["path"], [{"operation": "write_cell", "sheet": "Missing", "cell": "A1", "value": 1}], "outputs/bad.xlsx")
    assert invalid["status"] == "failure"


def test_workspace_path_restriction(tmp_path: Path):
    service = ArtifactService(tmp_path)
    denied = service.create_workbook({"worksheets": [{"name": "Data", "headers": ["A"], "rows": [[1]]}]}, "/etc/aegis.xlsx")
    assert denied["status"] == "failure"
    assert any("invalid_path" in error for error in denied["errors"])


def test_capability_routing_and_combined_decomposition():
    stages = decompose_task("Analyze this dataset and create both an Excel workbook and a PowerPoint summary")
    assert [stage.capability for stage in stages] == ["spreadsheet_generation", "presentation_generation"]

    # The master owns deterministic capability selection; no model call is needed.
    from models.registry import ModelRegistry
    registry = ModelRegistry.from_yaml("config/models.yaml")
    master = MasterAgent(__import__("runtime.agents", fromlist=["build_default_agent_registry"]).build_default_agent_registry(registry))
    assert master._capability_plan("Create a 5-slide PowerPoint about AEGIS")[0]["agent"] == "presentation_agent"
    assert master._capability_plan("Create an Excel budget workbook")[0]["agent"] == "spreadsheet_agent"


def test_symbolic_sum_routes_to_calculation_not_document():
    from models.registry import ModelRegistry
    from runtime.agents import build_default_agent_registry

    master = MasterAgent(build_default_agent_registry(ModelRegistry.from_yaml("config/models.yaml")))
    plan = master._capability_plan("whats sum of first n numbers")
    assert plan[0]["agent"] == "general_agent"
    assert plan[0]["capability"] == "calculation"


def test_ppt_shorthand_routes_to_presentation():
    from models.registry import ModelRegistry
    from runtime.agents import build_default_agent_registry

    master = MasterAgent(build_default_agent_registry(ModelRegistry.from_yaml("config/models.yaml")))
    plan = master._capability_plan("create a ppt about mlrp")
    assert plan[0]["agent"] == "presentation_agent"
    assert plan[0]["capability"] == "presentation_generation"


@pytest.mark.asyncio
async def test_master_artifact_path_combined_and_offline(tmp_path: Path):
    from models.registry import ModelRegistry
    from runtime.orchestrator import Orchestrator

    orchestrator = Orchestrator(ModelRegistry.from_yaml("config/models.yaml"), workspace_root=tmp_path)
    orchestrator._approval_fn = lambda *args: True
    result = await orchestrator.run_master("Analyze this dataset and create both an Excel workbook and a PowerPoint summary")
    assert result["status"] == "completed"
    assert result["verification"]["status"] == "verified"
    assert {Path(path).suffix for path in result["artifacts"]} == {".xlsx", ".pptx"}
    assert result["network_report"]["external_model_calls"] == 0
    assert result["network_report"]["external_tool_calls"] == 0

    denied = await orchestrator.run_master("Create an Excel workbook and save the file to /etc/test.xlsx")
    assert denied["status"] == "failed"
    assert not Path("/etc/test.xlsx").exists()


@pytest.mark.asyncio
async def test_artifact_failure_uses_bounded_repair(tmp_path: Path):
    from models.registry import ModelRegistry
    from runtime.orchestrator import Orchestrator

    orchestrator = Orchestrator(ModelRegistry.from_yaml("config/models.yaml"), workspace_root=tmp_path)
    orchestrator._approval_fn = lambda *args: True
    specialist = orchestrator.agent_registry.get("presentation_agent")
    original = specialist.tools["create_presentation"]
    calls = {"count": 0}

    def flaky_create(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"status": "failure", "artifact_type": "pptx", "errors": ["artifact_validation_failed"]}
        return original(**kwargs)

    specialist.tools["create_presentation"] = flaky_create
    result = await orchestrator.run_master("Create a 2-slide PowerPoint about AEGIS")
    assert result["status"] == "completed"
    assert calls["count"] == 2
    assert result["repair_history"]
    assert len(result["repair_history"]) <= 1

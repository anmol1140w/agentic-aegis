from pathlib import Path

from benchmarks.runner import BenchmarkCase, BenchmarkResult, summarize
from runtime.context_manager import ContextBudget, ContextBudgetManager
from runtime.model_profiles import get_model_profile
from runtime.repository_index import RepositoryIndex
from runtime.reviewer import deterministic_review
from tools.registry import ToolContract
from tools.workspace import WorkspaceReadTools


def test_context_budget_is_bounded_and_reports_truncation():
    bundle = ContextBudgetManager(ContextBudget(total_chars=100, code_chars=50)).build(
        system="system", task="task", files={"a.py": "x" * 200})
    assert bundle.prompt_chars <= 100
    assert bundle.truncated


def test_repository_index_extracts_python_symbols_and_imports(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("import os\n\nclass Example:\n    def run(self):\n        return os.getcwd()\n")
    index = RepositoryIndex(tmp_path, tmp_path / "index.db")
    info = index.refresh()
    assert info["symbols"] >= 2
    assert "sample.py" in index.relevant_files("Example")


def test_reviewer_rejects_missing_changed_file(tmp_path: Path):
    result = deterministic_review(workspace_root=str(tmp_path), result={"changed_files": ["missing.py"]})
    assert not result.passed


def test_profile_is_local_and_bounded():
    profile = get_model_profile()
    assert profile.local_only and profile.max_repairs == 3


def test_benchmark_summary_has_baseline_metrics():
    report = summarize([BenchmarkResult("x", "coder_only", success=True), BenchmarkResult("x", "aegis", success=True)])
    assert report["aegis"]["success_rate"] == 1.0


def test_tool_contract_is_structured():
    contract = ToolContract(name="read_file")
    assert contract.workspace_policy == "workspace_confined"


def test_namespaced_runtime_returns_stable_envelope_and_validates_input(tmp_path: Path):
    workspace = WorkspaceReadTools(tmp_path, approver=lambda *_: True)
    Path(tmp_path, "app.py").write_text("value = 1\n")
    result = workspace.invoke_tool("workspace.change.edit", {
        "path": "app.py", "find": "value = 1", "replace": "value = 2", "dry_run": True,
    }, request_id="req_test")
    assert result["ok"] and result["tool"] == "workspace.change.edit"
    assert result["request_id"] == "req_test"
    assert result["data"]["dry_run"] is True
    assert Path(tmp_path, "app.py").read_text() == "value = 1\n"


def test_namespaced_runtime_rejects_unknown_tool(tmp_path: Path):
    result = WorkspaceReadTools(tmp_path).invoke_tool("workspace.change.nope", {})
    assert not result["ok"] and result["error"]["code"] == "invalid_tool"

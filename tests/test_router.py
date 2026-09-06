"""Tests for task classification and model routing.

Verifies:
1. File/image modality detection from embedded paths in user requests.
2. Direct deterministic tool routing (calculator, list_files, read_file).
3. Tool vs Model routing distinctions.
4. Model capability scoring and fallback when models are unavailable.
5. Serialization safety of task representations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from models.base import ModelCapability, ModelConfig
from models.registry import ModelRegistry
from routing.classifier import (
    Complexity,
    ExecutionMode,
    Modality,
    Task,
    TaskClassifier,
    TaskType,
    extract_file_paths,
)
from routing.router import ModelRouter, RoutingResult
from routing.scoring import score_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier() -> TaskClassifier:
    return TaskClassifier()


@pytest.fixture
def sample_registry(tmp_path: Path) -> ModelRegistry:
    """Create a ModelRegistry from a temp YAML with known models."""
    yaml_content = """
models:
  test-general:
    provider: ollama
    model: "test-general:latest"
    capabilities:
      - general
      - reasoning
      - summarization
      - document_analysis
      - artifact_generation
    context_length: 32768
    priority: 10

  test-coder:
    provider: ollama
    model: "test-coder:latest"
    capabilities:
      - coding
      - debugging
      - calculation
    context_length: 32768
    priority: 10

  test-vision:
    provider: ollama
    model: "test-vision:latest"
    capabilities:
      - vision
      - multimodal
      - image_analysis
      - document_analysis
    context_length: 32768
    priority: 10

  test-fallback:
    provider: ollama
    model: "test-fallback:latest"
    capabilities:
      - general
      - lightweight
    context_length: 8192
    priority: 1
    is_fallback: true
"""
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml_content)
    return ModelRegistry.from_yaml(config_path)


@pytest.fixture
def router(sample_registry: ModelRegistry) -> ModelRouter:
    return ModelRouter(sample_registry)


# ---------------------------------------------------------------------------
# 1. Path Extraction & Image Modality Detection Tests
# ---------------------------------------------------------------------------

class TestPathAndModalityDetection:
    """Test extraction of paths from user request and modality detection."""

    def test_extract_absolute_image_path(self) -> None:
        paths = extract_file_paths("/path/to/image.jpg explain")
        assert "/path/to/image.jpg" in paths

    def test_extract_relative_png_path(self) -> None:
        paths = extract_file_paths("check ./diagram.png for errors")
        assert "./diagram.png" in paths

    def test_extract_quoted_path(self) -> None:
        paths = extract_file_paths('explain "my blueprint.webp" in detail')
        assert "my blueprint.webp" in paths

    def test_image_request_classifies_as_image_modality(self, classifier: TaskClassifier) -> None:
        # Prompt: /path/to/image.jpg explain
        task = classifier.classify("/path/to/image.jpg explain")
        assert task.modality == Modality.IMAGE
        assert task.requires_vision is True
        assert task.task_type == TaskType.IMAGE_ANALYSIS
        assert "/path/to/image.jpg" in task.attached_files

    def test_image_request_routes_to_vision_model(
        self, classifier: TaskClassifier, router: ModelRouter
    ) -> None:
        task = classifier.classify("/path/to/image.jpg explain this image")
        result = router.route(task)
        assert result.model_name == "test-vision"
        assert result.is_direct_tool is False

    def test_document_path_detection(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("read this document /tmp/inspection.pdf")
        assert task.modality == Modality.DOCUMENT
        assert task.requires_documents is True
        assert "/tmp/inspection.pdf" in task.attached_files


# ---------------------------------------------------------------------------
# 2. Direct Tool vs Model Routing Tests
# ---------------------------------------------------------------------------

class TestDirectToolRouting:
    """Test direct tool classification and execution routing."""

    def test_calculator_direct_tool(self, classifier: TaskClassifier, router: ModelRouter) -> None:
        task = classifier.classify("Calculate 234 * 918.")
        assert task.execution_mode == ExecutionMode.DIRECT_TOOL
        assert task.direct_tool_name == "calculator"
        assert "234 * 918" in task.direct_tool_args.get("expression", "")

        result = router.route(task)
        assert result.is_direct_tool is True
        assert result.model_name == "direct_tool"

    def test_pure_math_expression(self, classifier: TaskClassifier, router: ModelRouter) -> None:
        task = classifier.classify("25 * 37")
        assert task.execution_mode == ExecutionMode.DIRECT_TOOL
        assert task.direct_tool_name == "calculator"

        result = router.route(task)
        assert result.is_direct_tool is True

    def test_list_files_direct_tool(self, classifier: TaskClassifier, router: ModelRouter) -> None:
        task = classifier.classify("ls")
        assert task.execution_mode == ExecutionMode.DIRECT_TOOL
        assert task.direct_tool_name == "list_files"

        result = router.route(task)
        assert result.is_direct_tool is True
        assert result.model_name == "direct_tool"

    def test_read_file_direct_tool(self, classifier: TaskClassifier, router: ModelRouter) -> None:
        task = classifier.classify("read file workspace/report.txt")
        assert task.execution_mode == ExecutionMode.DIRECT_TOOL
        assert task.direct_tool_name == "read_file"
        assert task.direct_tool_args.get("file_path") == "workspace/report.txt"

    def test_create_directory_direct_tool(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("create directory ./output/logs")
        assert task.execution_mode == ExecutionMode.DIRECT_TOOL
        assert task.direct_tool_name == "create_directory"


# ---------------------------------------------------------------------------
# 3. Model Routing Acceptance Tests
# ---------------------------------------------------------------------------

class TestModelRoutingAcceptance:
    """Acceptance tests specified for model routing."""

    def test_general_binary_tree_request(
        self, classifier: TaskClassifier, router: ModelRouter
    ) -> None:
        # Prompt: Explain what a binary tree is.
        task = classifier.classify("Explain what a binary tree is.")
        assert task.task_type == TaskType.GENERAL
        assert task.execution_mode == ExecutionMode.MODEL
        result = router.route(task)
        assert result.model_name == "test-general"
        assert result.is_direct_tool is False

    def test_coding_sort_request(
        self, classifier: TaskClassifier, router: ModelRouter
    ) -> None:
        # Prompt: Write Python code to sort a list.
        task = classifier.classify("Write Python code to sort a list.")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL
        result = router.route(task)
        assert result.model_name == "test-coder"
        assert result.is_direct_tool is False

    def test_language_comparison_is_not_coding(self, classifier: TaskClassifier) -> None:
        assert classifier.classify("Is Python faster than JavaScript?").task_type == TaskType.GENERAL

    def test_python_function_request_is_coding(self, classifier: TaskClassifier) -> None:
        assert (
            classifier.classify(
                "Write a Python function that routes chunks by confidence_score"
            ).task_type
            == TaskType.CODING
        )

    @pytest.mark.parametrize("prompt", [
        "list all files in the current directory",
        "show files in this folder",
    ])
    def test_natural_language_list_directory_routes_to_workspace(self, classifier: TaskClassifier, prompt: str) -> None:
        task = classifier.classify(prompt)
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert task.requires_tools == ["workspace_read"]

    def test_natural_language_search_routes_to_workspace(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("find where document_analysis is routed and explain it")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS

    def test_natural_language_read_file_routes_to_workspace(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("open routing/classifier.py")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS

    def test_python_comparison_is_not_workspace_or_coding(self, classifier: TaskClassifier) -> None:
        assert classifier.classify("Is Python faster than JavaScript?").task_type == TaskType.GENERAL

    def test_coding_and_test_it_requests_sandbox_tool(
        self, classifier: TaskClassifier
    ) -> None:
        task = classifier.classify("Write Python code to calculate CRC and test it")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS

    def test_fallback_when_preferred_model_unavailable(
        self, classifier: TaskClassifier, router: ModelRouter
    ) -> None:
        task = classifier.classify("Write Python code to sort a list.")
        # Mark test-coder unavailable in availability map
        availability = {
            "test-coder": False,
            "test-general": False,
            "test-vision": False,
            "test-fallback": True,
        }
        result = router.route(task, availability=availability)
        assert result.model_name == "test-fallback"
        assert "fallback" in result.reason.lower()


# ---------------------------------------------------------------------------
# 4. Serialization Safety Tests
# ---------------------------------------------------------------------------

class TestSerializationSafety:
    """Verify task and trace produce primitive dicts for LangGraph checkpointing."""

    def test_task_to_serializable_dict_primitives(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("/path/to/img.png explain")
        d = task.to_serializable_dict()
        assert isinstance(d, dict)
        assert isinstance(d["modality"], str)
        assert d["modality"] == "image"
        assert isinstance(d["task_type"], str)
        assert isinstance(d["complexity"], str)
        assert isinstance(d["execution_mode"], str)
        assert isinstance(d["requires_vision"], bool)
        assert isinstance(d["attached_files"], list)
        # Ensure it contains NO Enum instances
        for val in d.values():
            assert not isinstance(val, type(Modality.TEXT))
            assert not isinstance(val, type(TaskType.GENERAL))
            assert not isinstance(val, type(Complexity.SIMPLE))
            assert not isinstance(val, type(ExecutionMode.MODEL))


# ---------------------------------------------------------------------------
# 5. Source-File vs Document Classification Regression Tests
# ---------------------------------------------------------------------------

class TestSourceClassification:
    """Verify source-code inspection routes to workspace and documents route to analysis."""

    def test_read_python_file_routes_to_workspace(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("read the verify_runs.py file")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert "workspace_read" in task.requires_tools
    def test_edit_python_file_routes_to_workspace(
        self, classifier: TaskClassifier
    ) -> None:
        task = classifier.classify(
            "edit verify_run.py and add one more testing in it"
        )

        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert "workspace_read" in task.requires_tools
        assert "edit_file" in task.requires_tools
        assert task.requires_human_approval is True


    def test_modify_python_file_routes_to_workspace(
        self, classifier: TaskClassifier
    ) -> None:
        task = classifier.classify("modify verify_run.py")

        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert "edit_file" in task.requires_tools
        assert task.requires_human_approval is True


    def test_add_test_to_python_file_routes_to_workspace(
        self, classifier: TaskClassifier
    ) -> None:
        task = classifier.classify("add one more test to verify_run.py")

        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert "edit_file" in task.requires_tools
        assert task.requires_human_approval is True

    def test_open_source_file_routes_to_workspace(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("open main.py")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert "workspace_read" in task.requires_tools

    def test_search_code_routes_to_workspace(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("search for NetworkMonitor")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert "workspace_read" in task.requires_tools

    def test_document_pdf_request_remains_document_analysis(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("analyze this inspection PDF")
        assert task.task_type == TaskType.DOCUMENT_ANALYSIS

    def test_inspection_report_request_remains_document_analysis(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("extract findings from this report")
        assert task.task_type == TaskType.DOCUMENT_ANALYSIS

    def test_python_vs_javascript_question_is_not_coding(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("Is Python faster than JavaScript?")
        assert task.task_type == TaskType.GENERAL

    def test_networkmonitor_search_routes_to_workspace(self, classifier: TaskClassifier) -> None:
        task = classifier.classify("find all Python files containing NetworkMonitor")
        assert task.task_type == TaskType.CODING
        assert task.execution_mode == ExecutionMode.MODEL_WITH_TOOLS
        assert "workspace_read" in task.requires_tools

import pytest

from routing.classifier import TaskClassifier, TaskType
from runtime.actions import ActionParseError, parse_action
from runtime.plan_parser import PlanValidationError, parse_plan
from runtime.regex_safety import bounded_text, compile_safe


def test_bounded_text_caps_regex_input() -> None:
    assert len(bounded_text("x" * 300_000)) == 256 * 1024


def test_malformed_or_oversized_patterns_are_rejected() -> None:
    with pytest.raises(ValueError):
        compile_safe("[")
    with pytest.raises(ValueError):
        compile_safe("a" * 3000)


def test_action_and_plan_parsers_bound_model_text() -> None:
    with pytest.raises(ActionParseError):
        parse_action("x" * 300_000)
    with pytest.raises(PlanValidationError):
        parse_plan("x" * 300_000, registered_tools=set())


def test_classifier_behavior_is_preserved_for_binary_tree() -> None:
    assert TaskClassifier().classify("Explain what a binary tree is.").task_type is TaskType.GENERAL

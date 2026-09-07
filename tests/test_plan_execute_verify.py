import pytest

from runtime.plan_parser import PlanValidationError, parse_plan
from runtime.self_healing import classify_failure, normalize_repair_action
from runtime.verification import verify_coding_result, verify_plan, verify_tool_result


def test_plan_parser_accepts_fenced_json_and_rejects_unknown_tools():
    plan = parse_plan(
        "```json\n{\"goal\":\"inspect\",\"steps\":[{\"step_id\":\"s1\",\"description\":\"read\",\"tool_name\":\"read_file\",\"parameters\":{\"path\":\"a.py\"}}]}\n```",
        registered_tools={"read_file"},
    )
    assert plan.steps[0].tool_name == "read_file"
    with pytest.raises(PlanValidationError, match="unknown tool"):
        parse_plan('{"goal":"bad","steps":[{"step_id":"s1","description":"x","tool_name":"rm"}]}',
                   registered_tools={"read_file"})


def test_verification_requires_real_command_evidence():
    assert verify_tool_result({"ok": True, "tool": "execute_command", "exit_code": 0, "timed_out": False})["passed"]
    assert not verify_tool_result({"ok": True, "tool": "execute_command", "exit_code": 0, "timed_out": True})["passed"]
    assert not verify_plan([{"ok": False, "tool": "execute_command", "exit_code": 1}])["passed"]


def test_coding_success_message_without_tool_evidence_is_not_verified():
    result = {"status": "success", "summary": "pytest passed", "metadata": {}}
    check = verify_coding_result(result, "fix the tests and run pytest")
    assert check["passed"] is False
    assert "actual execute_command evidence" in check["missing_evidence"]


def test_coding_pytest_exit_zero_is_verified_from_recorded_state():
    result = {
        "status": "success",
        "metadata": {"coding_state": {
            "commands": [{"command": "pytest tests/test_calculator.py", "exit_code": 0,
                           "stdout": "1 passed", "stderr": "", "state_version": 1}],
            "last_edit_state_version": 1,
            "verification": {"status": "passed"},
        }},
    }
    assert verify_coding_result(result, "repair the failing pytest tests")["passed"] is True


def test_self_healing_classifies_policy_failures_as_non_retryable():
    assert classify_failure({"error": "approval_denied"})["retryable"] is False
    assert classify_failure({"errors": ["approval_denied"]})["retryable"] is False
    assert classify_failure({"error": "command_timeout"})["retryable"] is True
    assert normalize_repair_action({"action": "retry", "parameters": {}})["action"] == "retry"

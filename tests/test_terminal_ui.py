from types import SimpleNamespace

from rich.console import Console

from ui.terminal import TerminalUI


class Registry:
    def list_models(self):
        return [SimpleNamespace(name="reasoner", model="local-reasoner:13b",
                                capabilities=[SimpleNamespace(value="reasoning")])]


def test_startup_and_models_use_registry_values_not_model_literals():
    console = Console(record=True, width=100, color_system=None)
    ui = TerminalUI(console)
    ui.startup(Registry(), {"reasoner": True})
    output = console.export_text()
    assert "1/1 available" in output
    ui.models(Registry(), {"reasoner": True})
    output = console.export_text()
    assert "local-reasoner:13b" in output
    assert "qwen" not in output.lower()


def test_event_view_hides_model_activity_and_shows_operational_state():
    console = Console(record=True, width=100, color_system=None)
    ui = TerminalUI(console)
    ui.event({"event": "model_activity", "agent": "reasoner", "token_count": 200})
    assert console.export_text() == ""
    ui.event({"event": "verification_passed"})
    assert "verification passed" in console.export_text()


def test_event_view_shows_sandbox_command_and_result():
    console = Console(record=True, width=120, color_system=None)
    ui = TerminalUI(console)
    ui.event({"event": "command_started", "command": "python -m py_compile fibonacci.py", "cwd": "workspace"})
    ui.event({"event": "command_finished", "exit_code": 0,
              "sandbox": {"backend": "process"}, "stdout_preview": "ok"})
    output = console.export_text()
    assert "python -m py_compile fibonacci.py" in output
    assert "cwd: workspace" in output
    assert "exit_code=0" in output
    assert "stdout › ok" in output


def test_result_view_keeps_route_and_verification_compact():
    console = Console(record=True, width=100, color_system=None)
    ui = TerminalUI(console)
    ui.result({"selected_model": "local-reasoner:13b",
               "task": {"required_capabilities": ["reasoning"]},
               "verification": {"status": "verified"}}, "done")
    output = console.export_text()
    assert "local-reasoner:13b" in output
    assert "verified" in output

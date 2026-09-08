#!/usr/bin/env python3
"""AEGIS — Sovereign Agent Workbench terminal client.

Interactive terminal interface for the agent runtime.
Streams LLM responses, executes deterministic tools directly,
shows task classification, model/tool routing, and detailed latency metrics.

Usage::

    python cli.py
    python cli.py inspect-report --input /path/to/report.pdf --output-dir workspace/outputs/run_001
    # or via entry point:
    sih-agent
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.theme import Theme

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
_WORKSPACE_ROOT = (_PROJECT_ROOT / "workspace").resolve()
_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.registry import ModelRegistry
from runtime.orchestrator import Orchestrator
from security.audit import AuditLogger
from security.network import NetworkMonitor
from tools.workspace import WorkspaceReadTools
from tools.files import set_workspace_root
from tools.artifacts import ArtifactError, ArtifactService
from ui.terminal import TerminalUI, AEGIS_THEME

# ---------------------------------------------------------------------------
# Rich theme
# ---------------------------------------------------------------------------

_THEME = AEGIS_THEME

console = Console(theme=_THEME)
terminal_ui = TerminalUI(console)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    console.set_window_title("AEGIS — Sovereign Agent Workbench")
    console.print("[accent]AEGIS[/accent]")
    console.print("[muted]Sovereign AI Workbench[/muted]")
    console.print(Rule(style="dim"))


def _print_network(network: NetworkMonitor) -> None:
    stats = network.snapshot()
    style = "success" if stats.external_connections == 0 else "error"
    console.print(
        f"\n[{style}]Agent network:[/{style}] "
        f"external={stats.external_connections}  local={stats.local_connections}  "
        f"observed_elsewhere_external={stats.observed_external_connections}  "
        f"model_calls={stats.local_model_calls}  "
        f"tool_calls={stats.local_tool_calls}"
    )


async def _handle_slash_command(raw_input: str, registry: ModelRegistry, network: NetworkMonitor) -> str | None:
    """Handle a terminal command exclusively, returning ``exit`` or ``handled``."""
    command = raw_input.strip().lower()
    if command in ("/quit", "/exit", "/q", "quit", "exit"):
        console.print("[info]Goodbye.[/info]")
        return "exit"
    if command == "/models":
        availability = await registry.check_availability()
        if hasattr(registry, "list_models"):
            terminal_ui.models(registry, availability)
        else:
            for name, available in availability.items():
                console.print(f"  {'✓' if available else '○'} {name}")
        return "handled"
    if command == "/network":
        _print_network(network)
        return "handled"
    if command.startswith("/validate-artifact") or command.startswith("/inspect-artifact"):
        parts = raw_input.strip().split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            console.print("[warning]Usage: /validate-artifact <workspace-relative .pptx/.xlsx path>[/warning]")
            return "handled"
        try:
            result = ArtifactService(_WORKSPACE_ROOT).validate(parts[1].strip())
            style = "success" if result.get("status") == "passed" else "error"
            console.print(Panel(
                f"[{style}]{result.get('status', 'failed').upper()}[/{style}]\n"
                f"Type: {result.get('artifact_type', 'unknown')}\n"
                f"Path: {parts[1].strip()}\n"
                f"Details: {result}",
                title="[bold bright_cyan]Artifact Validation[/bold bright_cyan]",
                border_style="green" if style == "success" else "red",
            ))
        except (ArtifactError, OSError) as exc:
            console.print(f"[error]Artifact validation denied: {exc}[/error]")
        return "handled"
    if command == "/artifacts":
        artifacts_root = _WORKSPACE_ROOT / "outputs" / "artifacts"
        files = sorted(path for path in artifacts_root.rglob("*") if path.is_file()) if artifacts_root.exists() else []
        if not files:
            console.print("[muted]No generated PPTX/XLSX artifacts found.[/muted]")
        else:
            for path in files[-30:]:
                console.print(f"  [success]✓[/success] {path.relative_to(_WORKSPACE_ROOT)} ({path.stat().st_size} bytes)")
        return "handled"
    if command == "/sandbox":
        # Probe the same canonical workspace used by the production agent.
        tools = WorkspaceReadTools(_WORKSPACE_ROOT, command_approver=lambda *_: True)
        status = tools.sandbox_status()
        if not status.get("ready"):
            console.print(Panel(
                f"[error]✗ Sandbox unavailable[/error]\n{status.get('message', 'No diagnostic available.')}",
                title="[bold bright_cyan]AEGIS Sandbox Check[/bold bright_cyan]", border_style="red"))
            return "handled"

        safe = tools.execute_command('python -c "print(2 + 2)"')
        blocked = tools.execute_command("curl http://example.com")
        safe_ok = safe.get("ok") is True and safe.get("exit_code") == 0 and safe.get("stdout", "").strip() == "4"
        blocked_ok = blocked.get("error") in {"disallowed_command", "capability_denied"}
        style = "success" if safe_ok and blocked_ok else "warning"
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim", width=20)
        table.add_column()
        table.add_row("Backend", str(status.get("backend", "unknown")))
        table.add_row("Workspace", str(status.get("workspace", _PROJECT_ROOT / "workspace")))
        table.add_row("Safe command", f"{'PASS' if safe_ok else 'FAIL'}  exit_code={safe.get('exit_code')}  stdout={safe.get('stdout', '').strip()!r}")
        table.add_row("Blocked command", f"{'PASS' if blocked_ok else 'FAIL'}  error={blocked.get('error', 'none')}")
        console.print(Panel(table, title="[bold bright_cyan]AEGIS Sandbox Check[/bold bright_cyan]",
                            border_style=style))
        return "handled"
    if command in {"/help", "/commands"}:
        terminal_ui.help()
        return "handled"
    if command == "/status":
        console.print(Panel(
            "[success]LOCAL[/success] inference via Ollama\n"
            "[success]MASTER[/success] is the only production orchestrator\n"
            "[success]POLICY[/success] guards paths, commands, approvals, and network\n"
            "[success]AUDIT[/success] records bounded operational evidence\n"
            "[dim]Private chain-of-thought is never displayed or persisted.[/dim]",
            title="[bold bright_cyan]AEGIS Status[/bold bright_cyan]", border_style="blue"))
        return "handled"
    if command == "/clear":
        console.clear()
        _print_banner()
        return "handled"
    if command == "/debug":
        terminal_ui.debug = not terminal_ui.debug
        console.print(f"[muted]debug output {'enabled' if terminal_ui.debug else 'disabled'}[/muted]")
        return "handled"
    if command == "/master":
        console.print("[info]Master-agent mode enabled for the next request.[/info]")
        return "master"
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def _run() -> None:
    # Resolve config paths
    config_dir = _PROJECT_ROOT / "config"
    models_yaml = config_dir / "models.yaml"

    if not models_yaml.exists():
        console.print(f"[error]Config not found: {models_yaml}[/error]")
        sys.exit(1)

    # Load model registry
    registry = ModelRegistry.from_yaml(models_yaml)

    # Health check
    console.print("[muted]Checking local model availability...[/muted]")
    availability = await registry.check_availability()
    any_available = any(availability.values())

    if not any_available:
        console.print(
            "\n[warning]No models are available in Ollama. "
            "Deterministic tools (calculator, file ops) will work directly. "
            "For LLM tasks, ensure 'ollama serve' is running.[/warning]"
        )

    # Initialise components
    set_workspace_root(_WORKSPACE_ROOT)
    audit = AuditLogger()
    network = NetworkMonitor()
    orchestrator = Orchestrator(
        registry,
        audit=audit,
        network=network,
        availability=availability,
        workspace_root=_WORKSPACE_ROOT,
    )
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    master_mode = False

    terminal_ui.startup(registry, availability)
    terminal_ui.session(thread_id)
    console.print(Rule("ready", style="dim"))

    try:
        while True:
            try:
                user_input = console.input("[cyan]you[/cyan] [dim]>[/dim] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[info]Goodbye.[/info]")
                break

            if not user_input:
                continue

            command_result = await _handle_slash_command(user_input, registry, network)
            if command_result == "exit":
                break
            if command_result == "handled":
                continue
            if command_result == "master":
                master_mode = True
                continue

            # Every request uses the same Master implementation. `/master` is only
            # an explicit alias, never a second execution path or fallback.
            if master_mode:
                master_mode = False
            if user_input.lower().startswith("/master "):
                user_input = user_input[8:].strip()

            try:
                def _master_progress(event: dict[str, Any]) -> None:
                    """Forward orchestration events to the centralized terminal UI."""
                    terminal_ui.event(event)

                terminal_ui.request(user_input)
                # Do not wrap the run in Rich's live spinner: approval prompts use
                # stdin and must remain visible/interactive on every terminal.
                master_result = await orchestrator.run_master(user_input, progress_callback=_master_progress)
                prep = master_result.get("preprocessing", {})
                if prep:
                    meta = prep.get("metadata", {})
                    console.print(
                        f"  [step]✓ NLP preprocessing:[/step] "
                        f"{meta.get('normalization_count', 0)} corrections, "
                        f"{meta.get('entity_count', 0)} entities, "
                        f"{meta.get('processing_duration_ms', 0):.1f}ms"
                    )
                raw_answer = str(master_result.get("final_answer", "Master could not complete the request."))
                # Sanitize: if the model returned a raw task-spec JSON blob as the
                # answer (keys like "operation", "original_request", etc.), replace
                # it with a human-readable fallback so the panel is readable.
                try:
                    import json as _j
                    _parsed = _j.loads(raw_answer.strip())
                    _machine = isinstance(_parsed, dict) and bool(
                        {"operation", "original_request", "normalized_request",
                         "agent", "domain_intent", "workflow"} & _parsed.keys()
                    )
                except Exception:
                    _machine = False
                if _machine:
                    _changes = master_result.get("agent_results", [{}])[-1].get("changes", []) if master_result.get("agent_results") else []
                    _arts = master_result.get("agent_results", [{}])[-1].get("artifacts", []) if master_result.get("agent_results") else []
                    if _changes:
                        answer = "Modified: " + ", ".join(str(p) for p in _changes[:6]) + "."
                    elif _arts:
                        answer = "Produced: " + ", ".join(str(a) for a in _arts[:4]) + "."
                    else:
                        answer = "Task completed. See output directory for artifacts."
                else:
                    answer = raw_answer
                terminal_ui.result(master_result, answer)
                console.print()
                continue
            except KeyboardInterrupt:
                # Ctrl-C cancels the active model/tool request and returns to the
                # terminal prompt.  It must not print an asyncio traceback or tear
                # down the whole REPL; this is the CLI equivalent of the reference
                # runtimes' abort signal.
                console.print("\n[warning]Request cancelled.[/warning] Returning to the terminal prompt.")
                continue
            except asyncio.CancelledError:
                console.print("\n[warning]Request cancelled.[/warning] Returning to the terminal prompt.")
                continue
            except Exception as exc:
                console.print(f"[error]Master execution failed: {exc}[/error]")
                continue

    finally:
        # Cleanup
        _print_network(network)
        audit.close()


def main() -> None:
    """Entry point for ``sih-agent`` CLI.

    Supports subcommands:
      - (no args): interactive REPL
      - inspect-report --input <path> [--output-dir <path>]: deterministic pipeline
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="aegis",
        description="AEGIS — Sovereign Agent Workbench terminal client",
    )
    subparsers = parser.add_subparsers(dest="command")

    # inspect-report subcommand
    ir_parser = subparsers.add_parser(
        "inspect-report",
        help="Run deterministic inspection-report pipeline",
    )
    ir_parser.add_argument(
        "--input", required=True,
        help="Path to scanned inspection report (PDF or image)",
    )
    ir_parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for run artifacts",
    )

    args = parser.parse_args()

    if args.command == "inspect-report":
        from pipeline.inspect_report import run_inspect_report
        from rich.panel import Panel as P

        console.print(P(
            "[bold blue]AEGIS Inspection Report Pipeline[/bold blue]\n"
            "[dim]Deterministic • Local • Offline[/dim]",
            border_style="blue",
        ))
        result = run_inspect_report(
            input_path=args.input,
            output_dir=args.output_dir,
        )
        sys.exit(0 if result.get("status") == "complete" else 1)
    else:
        asyncio.run(_run())


if __name__ == "__main__":
    main()

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
from rich.text import Text
from rich.theme import Theme

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.registry import ModelRegistry
from runtime.orchestrator import Orchestrator
from security.audit import AuditLogger
from security.network import NetworkMonitor
from tools.workspace import WorkspaceReadTools

# ---------------------------------------------------------------------------
# Rich theme
# ---------------------------------------------------------------------------

_THEME = Theme({
    "info": "dim cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "model": "bold magenta",
    "tool": "bold cyan",
    "step": "dim white",
})

console = Console(theme=_THEME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    console.set_window_title("AEGIS — Sovereign Agent Workbench")
    banner = Text()
    banner.append("AEGIS", style="bold bright_cyan")
    banner.append("  Sovereign Agent Workbench", style="bold blue")
    banner.append("\n")
    banner.append("Local inference  •  private workspace  •  policy-controlled tools", style="dim")
    console.print(Panel(banner, border_style="bright_cyan", padding=(1, 3)))


def _print_session_header(session_id: str, availability: dict[str, bool], registry: ModelRegistry) -> None:
    """Render a compact dashboard before entering the interactive REPL."""
    table = Table(show_header=True, header_style="bold bright_cyan", box=None, padding=(0, 2))
    table.add_column("Role", style="dim")
    table.add_column("Model")
    table.add_column("State", justify="center")
    for role in ("qwen-general", "qwen-vision", "qwen-coder", "llama-small"):
        try:
            model = registry.get_provider(role).config.model
        except Exception:
            model = "configured"
        available = availability.get(role, False)
        table.add_row(role, model, "[success]ONLINE[/success]" if available else "[error]OFFLINE[/error]")
    console.print(Panel(table, title=f"[bold]Session {session_id}[/bold]  [dim]Master-first / LangGraph[/dim]",
                        subtitle="[dim]Type /help for commands[/dim]", border_style="blue"))


def _print_result_summary(result: dict[str, Any]) -> None:
    status = str(result.get("status", "unknown"))
    style = "success" if status in {"completed", "success", "verified"} else "warning"
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=18)
    table.add_column()
    table.add_row("Status", f"[{style}]{status.upper()}[/{style}]")
    table.add_row("Specialist", result.get("selected_agent") or "—")
    table.add_row("Verification", str(result.get("verification", {}).get("status", "not run")))
    repairs = result.get("repair_history", [])
    table.add_row("Repairs", str(len(repairs)))
    if result.get("output_dir"):
        table.add_row("Run artifacts", str(result["output_dir"]))
    console.print(Panel(table, title="[bold bright_cyan]Run Summary[/bold bright_cyan]", border_style="bright_cyan"))


def _print_task_info(
    task_data: dict[str, Any] | Task,
    model_name: str,
    model_id: str,
    reason: str,
    is_direct_tool: bool = False,
) -> None:
    """Show classification, modality, routing, and execution mode."""
    if isinstance(task_data, Task):
        task_dict = task_data.to_serializable_dict()
    else:
        task_dict = task_data

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", width=16)
    table.add_column()

    table.add_row("Task type", f"[bold]{task_dict.get('task_type', 'general')}[/bold]")
    table.add_row("Modality", task_dict.get("modality", "text"))
    table.add_row("Execution mode", f"[bold]{task_dict.get('execution_mode', 'model')}[/bold]")

    if is_direct_tool:
        tool_name = task_dict.get("direct_tool_name", "tool")
        table.add_row("Direct tool", f"[tool]{tool_name}[/tool]")
    else:
        table.add_row("Model", f"[model]{model_name}[/model] ({model_id})")

    attached = task_dict.get("attached_files", [])
    if attached:
        table.add_row("Attached files", ", ".join(attached))

    tools = task_dict.get("requires_tools", [])
    if tools and not is_direct_tool:
        table.add_row("Tools bound", ", ".join(tools))

    console.print(Panel(table, title="[dim]Task Analysis & Routing[/dim]", border_style="dim", padding=(0, 1)))


def _print_trace(traces: list[dict[str, Any]], total_ms: float | None = None) -> None:
    """Show execution trace with phase timings."""
    for t in traces:
        dur = f" ({t['duration_ms']:.1f}ms)" if t.get("duration_ms") is not None else ""
        step = t.get("step", "")
        msg = t.get("message", "")
        console.print(f"  [step]✓ {step}:[/step] {msg}{dur}")

    if total_ms is not None:
        console.print(f"  [bold green]Total latency:[/bold green] [dim]{total_ms:.1f}ms[/dim]")


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
        for name, available in (await registry.check_availability()).items():
            console.print(f"  {'✓' if available else '✗'} {name}")
        return "handled"
    if command == "/network":
        _print_network(network)
        return "handled"
    if command == "/sandbox":
        # Probe the same project-root sandbox used by the production agent.
        tools = WorkspaceReadTools(_PROJECT_ROOT, command_approver=lambda *_: True)
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
        console.print(Panel(
            "[bold]/models[/bold]  show local model health\n"
            "[bold]/network[/bold] show network/tool counters\n"
            "[bold]/sandbox[/bold] verify command sandbox and policy\n"
            "[bold]/status[/bold]  show AEGIS operating principles\n"
            "[bold]/clear[/bold]   clear the terminal\n"
            "[bold]/quit[/bold]    exit AEGIS",
            title="[bold bright_cyan]AEGIS Commands[/bold bright_cyan]", border_style="blue"))
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
    if command == "/master":
        console.print("[info]Master-agent mode enabled for the next request.[/info]")
        return "master"
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def _run() -> None:
    _print_banner()

    # Resolve config paths
    config_dir = _PROJECT_ROOT / "config"
    models_yaml = config_dir / "models.yaml"

    if not models_yaml.exists():
        console.print(f"[error]Config not found: {models_yaml}[/error]")
        sys.exit(1)

    # Load model registry
    registry = ModelRegistry.from_yaml(models_yaml)

    # Health check
    console.print("\n[info]Checking model availability…[/info]")
    availability = await registry.check_availability()
    any_available = False
    for available in availability.values():
        if available:
            any_available = True

    if not any_available:
        console.print(
            "\n[warning]No models are available in Ollama. "
            "Deterministic tools (calculator, file ops) will work directly. "
            "For LLM tasks, ensure 'ollama serve' is running.[/warning]"
        )

    # Initialise components
    audit = AuditLogger()
    network = NetworkMonitor()
    orchestrator = Orchestrator(
        registry,
        audit=audit,
        network=network,
        availability=availability,
        workspace_root=_PROJECT_ROOT,
    )
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    master_mode = False

    _print_session_header(thread_id, availability, registry)
    console.print(Rule("Ready", style="dim blue"))

    while True:
        try:
            user_input = console.input("[bold green]You ▶[/bold green] ").strip()
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
                name = str(event.get("event", "")).replace("_", " ").title()
                if name in {"Preprocessing Started", "Preprocessing Completed"}:
                    return
                if name == "Master Plan":
                    console.print("  [step]… Master planning[/step]")
                elif name == "Capability Discovery":
                    console.print("  [step]✓ Capabilities discovered[/step]")
                elif name == "Plan Created":
                    console.print("  [step]… Master planning[/step]")
                elif name == "Plan Validated":
                    console.print("  [step]✓ Plan validated[/step]")
                elif name == "Specialist Execution":
                    console.print(f"  [step]⏳ {event.get('agent', 'specialist')} working…[/step]")
                elif name == "Model Activity":
                    # Coding-agent model output is an internal structured
                    # decision stream, not user-facing text. Keep the actual
                    # stream for action parsing, but do not flood the CLI with
                    # token-count progress lines.
                    if event.get("agent") == "coding_agent":
                        return
                    kind = event.get("kind", "inference")
                    console.print(f"  [dim cyan]⋯ {event.get('agent', 'model')} streaming {kind} ({event.get('token_count', 0)} chars)[/dim cyan]")
                elif name == "Step Succeeded":
                    console.print("  [step]✓ Specialist result recorded[/step]")
                elif name == "Verification Passed":
                    console.print("  [step]✓ Deterministic verification passed[/step]")
                elif name == "Verification Failed":
                    console.print("  [warning]⚠ Deterministic verification failed[/warning]")
                elif name == "Repair Requested":
                    console.print("  [warning]↻ Master applying bounded repair[/warning]")
                elif name == "Repair Rejected":
                    console.print("  [warning]⚠ No compatible repair available[/warning]")
                elif name == "Run Failed":
                    console.print("  [error]✗ Task failed safely[/error]")
                elif name == "Sandbox Preflight":
                    console.print("  [dim white]◇ sandbox preflight: checking local execution boundary…[/dim white]")
                elif name == "Command Started":
                    console.print(f"  [bright_white]▶ command running inside sandbox[/bright_white]  [dim]{event.get('command', '')}[/dim]")
                elif name == "Command Finished":
                    exit_code = event.get("exit_code")
                    state_style = "dim white" if event.get("status") == "success" and exit_code == 0 else "warning"
                    console.print(f"  [{state_style}]■ command stopped[/{state_style}]  exit_code={exit_code}  sandbox={event.get('sandbox', {}).get('status', 'unknown') if isinstance(event.get('sandbox'), dict) else 'unknown'}")
                    stdout = str(event.get("stdout_preview", "")).strip()
                    stderr = str(event.get("stderr_preview", "")).strip()
                    if stdout:
                        console.print(f"    [dim white]stdout › {stdout[:240]}[/dim white]")
                    if stderr:
                        console.print(f"    [dim yellow]stderr › {stderr[:240]}[/dim yellow]")
                elif name == "Command Output":
                    channel = event.get("stream", "stdout")
                    style = "dim white" if channel == "stdout" else "dim yellow"
                    text = str(event.get("text", "")).rstrip()
                    if text:
                        console.print(f"    [{style}]{channel} › {text[:240]}[/{style}]")
                elif name == "Master Delegate":
                    console.print(f"  [step]→ Delegating to {event.get('agent', 'specialist')}[/step]")
                elif name == "Agent Start":
                    console.print(f"  [step]⏳ {event.get('agent', 'specialist')} working…[/step]")
                elif name == "Agent Complete":
                    console.print(f"  [step]✓ {event.get('agent', 'specialist')} completed[/step]")
                elif name == "Tool Requested":
                    tool_name = event.get("tool", "tool")
                    args = event.get("arguments", {})
                    summary = " ".join(f"{key}={value}" for key, value in args.items())
                    console.print(f"  [tool]→ {tool_name}[/tool]" + (f"  [dim]{summary[:240]}[/dim]" if summary else ""))
                elif name == "Tool Result":
                    status = str(event.get("status", "unknown"))
                    style = "success" if status == "success" else "warning"
                    detail = f"exit_code={event.get('exit_code')}" if event.get("exit_code") is not None else status
                    console.print(f"  [{style}]■ {event.get('tool', 'tool')} {detail}[/{style}]")
                    stderr = str(event.get("stderr_preview", "")).strip()
                    if stderr:
                        console.print(f"    [dim yellow]stderr › {stderr[:240]}[/dim yellow]")
                elif name == "Repair Requested":
                    console.print(f"  [warning]↻ {event.get('agent', 'agent')} diagnosing failure and replanning[/warning]")
                elif name == "Master Review":
                    console.print("  [step]✓ Master reviewing result[/step]")
                elif name == "Master Replan":
                    console.print("  [warning]↻ Master replanning after specialist result[/warning]")
                elif name == "Final":
                    console.print("  [step]✓ Final response prepared[/step]")

            console.print()
            console.print(Panel(f"[bold]Request[/bold]\n{user_input}", border_style="dim", padding=(0, 1)))
            console.print("  [info]⏳ Local model processing…[/info]")
            # Do not wrap the run in Rich's live spinner: approval prompts use
            # stdin and must remain visible/interactive on every terminal.
            def _live_progress(event: dict[str, Any]) -> None:
                _master_progress(event)
            master_result = await orchestrator.run_master(user_input, progress_callback=_live_progress)
            prep = master_result.get("preprocessing", {})
            if prep:
                meta = prep.get("metadata", {})
                console.print(
                    f"  [step]✓ NLP preprocessing:[/step] "
                    f"{meta.get('normalization_count', 0)} corrections, "
                    f"{meta.get('entity_count', 0)} entities, "
                    f"{meta.get('processing_duration_ms', 0):.1f}ms"
                )
            answer = str(master_result.get("final_answer", "Master could not complete the request."))
            console.print(Panel(answer, title="[bold bright_cyan]AEGIS Response[/bold bright_cyan]", border_style="green" if not master_result.get("errors") else "yellow"))
            if master_result.get("errors"):
                console.print(f"[warning]Master errors: {master_result['errors']}[/warning]")
            if master_result.get("output_dir"):
                console.print(f"[success]✓ output saved:[/success] {master_result['output_dir']}")
            _print_result_summary(master_result)
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

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
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.registry import ModelRegistry
from routing.classifier import Task
from runtime.orchestrator import Orchestrator
from security.audit import AuditLogger
from security.network import NetworkMonitor

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
    banner.append("AEGIS — Sovereign Agent Workbench", style="bold blue")
    banner.append("\n")
    banner.append("Local • Private • Secure", style="dim")
    console.print(Panel(banner, border_style="blue", padding=(1, 2)))


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
    for name, available in availability.items():
        icon = "✓" if available else "✗"
        style = "success" if available else "error"
        cfg = registry.get_provider(name).config
        console.print(f"  [{style}]{icon}[/{style}] {name}  →  {cfg.model}")
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
    )
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    master_mode = False

    console.print(
        f"\n[info]Session: {thread_id}[/info]"
        f"\n[info]Commands: /quit /models /network[/info]\n"
    )

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

        # Opt-in master entry point. Legacy classifier/router remains the
        # default path and continues to provide the emergency fallback.
        if master_mode or user_input.lower().startswith("/master "):
            master_request = user_input[8:].strip() if user_input.lower().startswith("/master ") else user_input
            master_mode = False
            try:
                result = await orchestrator.run_master(master_request)
                answer = str(result.get("final_answer", ""))
                console.print(answer or "[warning]Master completed without a final answer.[/warning]")
                if result.get("errors"):
                    console.print(f"[warning]Master notes: {result['errors']}[/warning]")
            except Exception as exc:
                console.print(f"[error]Master error; legacy path remains available: {exc}[/error]")
            console.print()
            continue

        # Master-only default path. Failures are reported as structured Master
        # results; no alternate router is invoked.
        try:
            def _master_progress(event: dict[str, Any]) -> None:
                name = str(event.get("event", "")).replace("_", " ").title()
                if name in {"Preprocessing Started", "Preprocessing Completed"}:
                    return
                if name == "Master Plan":
                    console.print("  [step]… Master planning[/step]")
                elif name == "Capability Discovery":
                    console.print("  [step]✓ Capabilities discovered[/step]")
                elif name == "Master Delegate":
                    console.print(f"  [step]→ Delegating to {event.get('agent', 'specialist')}[/step]")
                elif name == "Agent Start":
                    console.print(f"  [step]⏳ {event.get('agent', 'specialist')} working…[/step]")
                elif name == "Agent Complete":
                    console.print(f"  [step]✓ {event.get('agent', 'specialist')} completed[/step]")
                elif name == "Master Review":
                    console.print("  [step]✓ Master reviewing result[/step]")
                elif name == "Master Replan":
                    console.print("  [warning]↻ Master replanning after specialist result[/warning]")
                elif name == "Final":
                    console.print("  [step]✓ Final response prepared[/step]")

            console.print("  [info]⏳ Local model processing…[/info]")
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
            answer = str(master_result.get("final_answer", "Master could not complete the request."))
            console.print(answer)
            if master_result.get("errors"):
                console.print(f"[warning]Master errors: {master_result['errors']}[/warning]")
            if master_result.get("output_dir"):
                console.print(f"[success]✓ output saved:[/success] {master_result['output_dir']}")
            console.print()
            continue
        except Exception as exc:
            console.print(f"[error]Master execution failed: {exc}[/error]")
            continue

        # -- Run the agent --
        task_info: dict[str, Any] | None = None
        model_name = ""
        model_id = ""
        routing_reason = ""
        is_direct_tool = False
        response_text = ""
        printed_length = 0
        traces: list[dict[str, Any]] = []
        printed_header = False
        total_ms: float | None = None
        import re

        try:
            async for event in orchestrator.astream(user_input, thread_id):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # -- Capture state updates from node outputs --
                if kind == "on_chain_end" and name in ("classify", "route", "execute_tool", "execute_model"):
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        if "task" in output and output["task"] is not None:
                            task_info = output["task"]
                        if "selected_model" in output:
                            model_name = output["selected_model"]
                        if "selected_model_id" in output:
                            model_id = output["selected_model_id"]
                        if "routing_reason" in output:
                            routing_reason = output["routing_reason"]
                        if "is_direct_tool" in output:
                            is_direct_tool = output["is_direct_tool"]
                        if "total_ms" in output:
                            total_ms = output["total_ms"]
                        if "trace" in output and output["trace"]:
                            for entry in output["trace"]:
                                if entry not in traces:
                                    traces.append(entry)

                        # Print task info after routing completes
                        if name == "route" and task_info and not printed_header:
                            _print_task_info(
                                task_info,
                                model_name,
                                model_id,
                                routing_reason,
                                is_direct_tool=is_direct_tool,
                            )
                            if not is_direct_tool:
                                console.print(f"[MODEL] {model_name} started")
                            else:
                                console.print()
                            printed_header = True

                        # For direct tool execution (which doesn't stream chat tokens), print result directly
                        if name == "execute_tool" and "messages" in output:
                            tool_msg = output["messages"][-1]
                            tool_content = (
                                tool_msg.content if hasattr(tool_msg, "content") else str(tool_msg)
                            )
                            console.print(tool_content)
                            response_text = tool_content

                        if name == "execute_model" and output.get("current_step") == "error":
                            model_content = str(output["messages"][-1].content)
                            console.print(f"[error]{model_content}[/error]")
                            response_text = model_content

                        if name in ("execute_tool", "execute_model") and output.get("output_dir"):
                            console.print(
                                f"\n[success]✓ output saved:[/success]\n  "
                                f"{Path(output['output_dir']) / 'result.txt'}"
                            )
                            if not is_direct_tool:
                                console.print(f"[MODEL] completed")

                if kind == "on_agent_status":
                    if name == "tool":
                        console.print(f"[TOOL] {data.get('tool', 'unknown')}")
                    elif name == "tool_result":
                        console.print(f"[step]✓ tool result received[/step]")
                    elif name == "thinking" and os.getenv("SHOW_OLLAMA_THINKING", "0") == "1":
                        # Opt-in, clearly separated, and never persisted as the final answer.
                        print(f"\r[thinking] {str(data.get('token', ''))[:500]}", end="", flush=True)
                    elif data.get("message"):
                        console.print(f"[warning]{data['message']}[/warning]")

                # -- Stream tokens from the LLM (for model tasks) --
                if kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        # Stream only the visible model response. Never display provider reasoning fields.
                        response_token = ""
                        if hasattr(chunk, "content") and chunk.content:
                            response_token = chunk.content
                        elif isinstance(chunk, dict):
                            response_token = chunk.get("content") or chunk.get("response") or ""

                        if response_token:
                            response_text += response_token
                            clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
                            clean_text = re.sub(r"<think>.*", "", clean_text, flags=re.DOTALL)
                            if len(clean_text) > printed_length:
                                print(clean_text[printed_length:], end="", flush=True)
                                printed_length = len(clean_text)

            # Newline after streamed response
            if response_text and not is_direct_tool:
                print()

            # Show trace & timings
            if traces:
                console.print()
                _print_trace(traces, total_ms=total_ms)

        except KeyboardInterrupt:
            console.print("\n[warning]Interrupted.[/warning]")
        except Exception as exc:
            console.print(f"\n[error]Error: {exc}[/error]")

        console.print()  # spacing between turns

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

"""Restrained terminal presentation for the AEGIS runtime."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.theme import Theme


AEGIS_THEME = Theme({
    "primary": "white",
    "muted": "dim",
    "accent": "bold cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "model": "cyan",
    "tool": "white",
})


def _status(ok: bool, true: str = "online", false: str = "offline") -> str:
    return f"[green]● {true}[/green]" if ok else f"[dim]○ {false}[/dim]"


class TerminalUI:
    """Owns terminal rendering only; it never executes runtime operations."""

    def __init__(self, console: Console | None = None, *, debug: bool = False) -> None:
        self.console = console or Console(theme=AEGIS_THEME)
        self.debug = debug
        self._streaming = False

    def startup(self, registry: Any, availability: dict[str, bool]) -> None:
        models = registry.list_models()
        available = sum(1 for name in availability if availability[name])
        rows = [
            ("LOCAL RUNTIME", "Ollama"),
            ("MODELS", f"{available}/{len(models)} available"),
            ("POLICY", "enforced"),
            ("SANDBOX", "isolated"),
            ("NETWORK", "restricted"),
            ("STATUS", "ready"),
        ]
        body = "\n".join(f"[dim]{label:<16}[/dim] {value}" for label, value in rows)
        self.console.print("[cyan]AEGIS[/cyan]")
        self.console.print("[dim]Sovereign AI Workbench[/dim]")
        self.console.print(Rule(style="dim"))
        self.console.print(body)
        if self.debug:
            self.console.print("\n[dim]Configured models[/dim]")
            self.models(registry, availability)
        self.console.print(Rule(style="dim"))

    def models(self, registry: Any, availability: dict[str, bool]) -> None:
        """Render model names from the registry; no provider IDs are hard-coded."""
        table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
        table.add_column("role")
        table.add_column("model")
        table.add_column("capabilities")
        table.add_column("state")
        for config in registry.list_models():
            table.add_row(config.name, config.model,
                          ", ".join(cap.value for cap in config.capabilities),
                          _status(bool(availability.get(config.name, False))))
        self.console.print(table)

    def session(self, session_id: str) -> None:
        self.console.print(f"[cyan]AEGIS[/cyan]  [dim]/ session {session_id}[/dim]")

    def request(self, text: str) -> None:
        self.console.print()
        self.console.print(f"[dim]>[/dim] {text}")

    def event(self, event: dict[str, Any]) -> None:
        """Render safe operational events, excluding model reasoning."""
        name = str(event.get("event", "")).replace("_", " ").lower()
        if name in {"preprocessing started", "preprocessing completed"}:
            return
        if name == "model activity":
            kind = str(event.get("kind", "content"))
            if kind == "thinking":
                if not self._streaming:
                    self.console.print("[dim cyan]thinking…[/dim cyan]", end=" ")
                    self._streaming = True
                return
            token = str(event.get("text", ""))
            if token:
                if not self._streaming:
                    self.console.print("[dim]response ›[/dim] ", end="")
                    self._streaming = True
                self.console.print(token, end="")
            return
        if name in {"agent complete", "final", "run failed"} and self._streaming:
            self.console.print()
            self._streaming = False
        if name == "command started":
            command = str(event.get("command", "")).strip()
            cwd = str(event.get("cwd", "")).strip()
            self.console.print(f"[white]▶ sandbox[/white] [bold]{command[:500]}[/bold]")
            if cwd:
                self.console.print(f"  [dim]cwd: {cwd[:240]}[/dim]")
            return
        if name == "command finished":
            exit_code = event.get("exit_code")
            marker = "✓" if exit_code == 0 else "×"
            style = "green" if exit_code == 0 else "red"
            sandbox = event.get("sandbox") or {}
            backend = sandbox.get("backend") if isinstance(sandbox, dict) else None
            suffix = f"  sandbox={backend}" if backend else ""
            self.console.print(f"[{style}]{marker}[/{style}] [white]sandbox finished[/white] exit_code={exit_code}{suffix}")
            for label, key in (("stdout", "stdout_preview"), ("stderr", "stderr_preview")):
                value = str(event.get(key, "")).strip()
                if value:
                    self.console.print(f"  [dim]{label} › {value[:500]}[/dim]")
            return
        mapping = {
            "master plan": ("•", "planning"),
            "plan created": ("•", "plan created"),
            "plan validated": ("✓", "plan validated"),
            "policy gateway": ("✓", "policy checked"),
            "approval checkpoint": ("!", "waiting for approval"),
            "checkpoint persisted": ("✓", "checkpoint saved"),
            "specialist execution": ("•", f"{event.get('agent', 'specialist')} working"),
            "step succeeded": ("✓", "step completed"),
            "verification passed": ("✓", "verification passed"),
            "verification failed": ("!", "verification failed"),
            "repair requested": ("•", "recovering from failure"),
            "repair rejected": ("!", "recovery unavailable"),
            "run failed": ("×", "execution failed safely"),
            "sandbox preflight": ("•", "checking isolated sandbox"),
            "command started": ("•", "running command in sandbox"),
            "command finished": ("✓" if event.get("exit_code") == 0 else "×", "command finished"),
            "master delegate": ("→", f"delegating to {event.get('agent', 'specialist')}"),
            "agent start": ("•", f"{event.get('agent', 'specialist')} started"),
            "agent complete": ("✓", f"{event.get('agent', 'specialist')} completed"),
            "tool requested": ("→", f"tool: {event.get('tool', 'unknown')}"),
            "tool result": ("✓" if event.get("status") == "success" else "!", f"tool: {event.get('tool', 'unknown')}"),
            "master review": ("•", "reviewing result"),
            "master replan": ("•", "replanning after result"),
            "final": ("✓", "response prepared"),
        }
        marker, message = mapping.get(name, ("•", name or "working"))
        style = "green" if marker == "✓" else "yellow" if marker == "!" else "red" if marker == "×" else "white"
        self.console.print(f"[{style}]{marker}[/{style}] {message}")
        if self.debug and name in {"command finished", "tool result"}:
            detail = event.get("stderr_preview") or event.get("exit_code") or event.get("status")
            if detail:
                self.console.print(f"  [dim]{str(detail)[:240]}[/dim]")

    def result(self, result: dict[str, Any], answer: str) -> None:
        errors = result.get("errors") or []
        self.console.print(Panel(answer, title="[cyan]Response[/cyan]", border_style="red" if errors else "green", padding=(0, 1)))
        task = result.get("task") or {}
        route = result.get("selected_model") or result.get("selected_model_id") or "local"
        capability = ", ".join(map(str, task.get("required_capabilities", []))) or "general"
        verification = result.get("verification") or {}
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim", width=16)
        table.add_column()
        table.add_row("route", capability)
        table.add_row("model", str(route))
        table.add_row("verification", verification.get("status", "not run"))
        table.add_row("approval", "required" if task.get("requires_human_approval") else "not required")
        if result.get("output_dir"):
            table.add_row("artifacts", str(result["output_dir"]))
        if errors:
            table.add_row("error", str(errors[0])[:240])
        self.console.print(table)

    def help(self) -> None:
        self.console.print("[cyan]Commands[/cyan]\n"
                          "  /models   configured model IDs and health\n"
                          "  /network  local network and tool counters\n"
                          "  /sandbox  sandbox preflight and policy check\n"
                          "  /status   operating policy\n"
                          "  /debug    toggle detailed operational output\n"
                          "  /clear    clear the terminal\n"
                          "  /quit     exit AEGIS")

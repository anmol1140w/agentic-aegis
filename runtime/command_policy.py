"""Policy layer for controlled command execution in workspace."""

from __future__ import annotations

import re
import shlex
import ast
from dataclasses import dataclass
from pathlib import Path

DENIED_COMMANDS: set[str] = {
    "sudo", "shutdown", "reboot", "mkfs", "dd", "fdisk", "parted",
    "ip", "ifconfig", "curl", "wget", "nc", "netcat", "bash", "sh",
    "zsh", "eval", "exec", "systemctl", "service", "kill", "pkill",
    "ssh", "rm",
}

# Disallow shell operators (chaining, backgrounding, redirection, subshells)
SHELL_OPERATORS: list[str] = [";", "&&", "||", "|", "&", ">", "<", "`", "$("]

READONLY_COMMANDS: set[str] = {
    "pwd", "ls", "find", "grep", "cat", "head", "tail",
}

MUTATING_COMMANDS: set[str] = {
    "mkdir", "cp", "mv",
}


@dataclass(frozen=True)
class CommandPolicyDecision:
    allowed: bool
    requires_approval: bool
    reason: str
    command_name: str = ""


def _has_unquoted_shell_operator(command: str) -> bool:
    """Reject shell syntax while allowing quoted Python source text."""
    quote = ""
    escaped = False
    for char in command:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in {";", "|", "&", ">", "<", "`", "$"}:
            return True
    return False


def evaluate_command(
    command: str,
    cwd: Path,
    workspace_root: Path,
) -> CommandPolicyDecision:
    """Evaluate command against local security policy.
    
    Returns CommandPolicyDecision indicating whether command is allowed,
    requires explicit approval, or is denied.
    """
    cmd_str = (command or "").strip()
    if not cmd_str:
        return CommandPolicyDecision(
            allowed=False, requires_approval=False, reason="Empty command"
        )

    # Reject shell chaining, piping, redirection, command substitution
    if _has_unquoted_shell_operator(cmd_str):
        return CommandPolicyDecision(
            allowed=False,
            requires_approval=False,
            reason="Shell chaining, piping, and redirection are not permitted",
        )

    try:
        tokens = shlex.split(cmd_str)
    except ValueError as exc:
        return CommandPolicyDecision(
            allowed=False, requires_approval=False, reason=f"Command parsing error: {exc}"
        )

    if not tokens:
        return CommandPolicyDecision(
            allowed=False, requires_approval=False, reason="Empty command tokens"
        )

    first = tokens[0].lower()
    base_first = Path(first).name.lower()

    # Explicitly denied binaries
    if first in DENIED_COMMANDS or base_first in DENIED_COMMANDS:
        return CommandPolicyDecision(
            allowed=False,
            requires_approval=False,
            reason=f"Command '{first}' is strictly denied by security policy",
            command_name=first,
        )

    # Check argument paths for escape attempts
    for arg in tokens[1:]:
        if arg.startswith("-"):
            continue
        p = Path(arg)
        if p.is_absolute():
            try:
                p.resolve().relative_to(workspace_root.resolve())
            except ValueError:
                return CommandPolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=f"Path argument '{arg}' resolves outside workspace",
                    command_name=first,
                )
        elif ".." in p.parts:
            resolved = (cwd / p).resolve()
            try:
                resolved.relative_to(workspace_root.resolve())
            except ValueError:
                return CommandPolicyDecision(
                    allowed=False,
                    requires_approval=False,
                    reason=f"Path argument '{arg}' traverses outside workspace",
                    command_name=first,
                )

    # Check pytest commands
    # Matches: pytest, .venv/bin/pytest, python -m pytest, .venv/bin/python -m pytest
    if base_first == "pytest":
        return CommandPolicyDecision(
            allowed=True,
            requires_approval=False,
            reason="Allowed read-only test command",
            command_name="pytest",
        )

    if base_first in ("python", "python3") and len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
        return CommandPolicyDecision(
            allowed=True,
            requires_approval=False,
            reason="Allowed read-only test command",
            command_name="pytest",
        )

    # Narrow process-observation probe used by the terminal sandbox smoke test.
    # It exposes only PID/PPID/time and cannot read files or spawn processes.
    if base_first in ("python", "python3") and len(tokens) == 3 and tokens[1] == "-c":
        source = tokens[2]
        try:
            tree = ast.parse(source, mode="exec")
            allowed_nodes = (ast.Module, ast.Import, ast.alias, ast.Expr, ast.Call,
                             ast.Name, ast.Attribute, ast.Load, ast.JoinedStr,
                             ast.FormattedValue, ast.Constant)
            if not all(isinstance(node, allowed_nodes) for node in ast.walk(tree)):
                raise ValueError("only process-observation expressions are allowed")
            imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
            imported_names = {alias.name for item in imports for alias in item.names}
            if len(imports) != 1 or not imported_names or not imported_names.issubset({"os", "time"}):
                raise ValueError("only os and time may be imported")
            attributes = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
            allowed_attributes = {("os", "getpid"), ("os", "getppid"), ("time", "time_ns")}
            for attribute in attributes:
                if not isinstance(attribute.value, ast.Name) or (attribute.value.id, attribute.attr) not in allowed_attributes:
                    raise ValueError("unsupported process-observation attribute")
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            allowed_calls = {("os", "getpid"), ("os", "getppid"), ("time", "time_ns"), ("", "print")}
            for call in calls:
                if call.keywords:
                    raise ValueError("keyword arguments are not allowed")
                if isinstance(call.func, ast.Name):
                    key = ("", call.func.id)
                elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                    key = (call.func.value.id, call.func.attr)
                else:
                    raise ValueError("unsupported function")
                if key not in allowed_calls:
                    raise ValueError("unsupported process-observation call")
            if not any(isinstance(call.func, ast.Name) and call.func.id == "print" for call in calls):
                raise ValueError("expected a process-observation print")
            if not any(isinstance(call.func, ast.Attribute)
                       and isinstance(call.func.value, ast.Name)
                       and call.func.value.id == "os" and call.func.attr == "getpid"
                       for call in calls):
                raise ValueError("expected os.getpid()")
        except (SyntaxError, ValueError):
            pass
        else:
            return CommandPolicyDecision(True, False, "Allowed process-observation probe", command_name="python")

    # Narrow deterministic arithmetic only; no imports, filesystem, dynamic
    # evaluation, attributes, assignments, or subprocess/network access.
    if base_first in ("python", "python3") and len(tokens) == 3 and tokens[1] == "-c":
        source = tokens[2]
        try:
            tree = ast.parse(source, mode="exec")
            allowed_nodes = (ast.Module, ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
                             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                             ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Load,
                             ast.Expr, ast.Call, ast.Name)
            if not all(isinstance(node, allowed_nodes) for node in ast.walk(tree)):
                raise ValueError("only arithmetic expressions are allowed")
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            if any(not isinstance(node.func, ast.Name) or node.func.id != "print" for node in calls):
                raise ValueError("only print(arithmetic expression) is allowed")
            if len(calls) > 1 or any(len(node.args) != 1 or node.keywords for node in calls):
                raise ValueError("only one print argument is allowed")
            if not calls and not any(isinstance(node, ast.BinOp) for node in ast.walk(tree)):
                raise ValueError("an arithmetic expression is required")
            if any(isinstance(node, ast.Constant) and not isinstance(node.value, (int, float))
                   for node in ast.walk(tree)):
                raise ValueError("only numeric constants are allowed")
        except (SyntaxError, ValueError) as exc:
            return CommandPolicyDecision(False, False, f"Unsafe python -c expression: {exc}", command_name="python")
        return CommandPolicyDecision(True, False, "Allowed deterministic arithmetic", command_name="python")

    # Workspace-local Python files may be run, but remain approval-gated.
    if base_first in ("python", "python3") and len(tokens) >= 2 and tokens[1].lower().endswith(".py"):
        script = Path(tokens[1])
        target = (cwd / script).resolve() if not script.is_absolute() else script.resolve()
        try:
            target.relative_to(workspace_root.resolve())
        except ValueError:
            return CommandPolicyDecision(False, False, "Python script is outside workspace", command_name="python")
        return CommandPolicyDecision(True, True, "Workspace-local Python execution requires approval", command_name="python")

    # Git commands
    if base_first == "git" and len(tokens) >= 2:
        subcmd = tokens[1].lower()
        if subcmd in ("status", "diff"):
            return CommandPolicyDecision(
                allowed=True,
                requires_approval=False,
                reason="Allowed read-only git command",
                command_name=f"git {subcmd}",
            )
        if subcmd in ("add", "commit"):
            return CommandPolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Mutating git command requires approval",
                command_name=f"git {subcmd}",
            )
        return CommandPolicyDecision(
            allowed=False,
            requires_approval=False,
            reason=f"Git subcommand '{subcmd}' is not permitted",
            command_name=f"git {subcmd}",
        )

    # Pip commands
    if base_first in ("pip", "pip3") and len(tokens) >= 2:
        subcmd = tokens[1].lower()
        if subcmd == "install":
            return CommandPolicyDecision(
                allowed=True,
                requires_approval=True,
                reason="Package installation requires approval",
                command_name="pip install",
            )
        return CommandPolicyDecision(
            allowed=False,
            requires_approval=False,
            reason=f"Pip subcommand '{subcmd}' is not permitted",
            command_name=f"pip {subcmd}",
        )

    # Read-only standard utilities
    if base_first in READONLY_COMMANDS:
        return CommandPolicyDecision(
            allowed=True,
            requires_approval=False,
            reason="Allowed read-only utility",
            command_name=base_first,
        )

    # Mutating utilities
    if base_first in MUTATING_COMMANDS:
        return CommandPolicyDecision(
            allowed=True,
            requires_approval=True,
            reason=f"Mutating command '{base_first}' requires approval",
            command_name=base_first,
        )

    return CommandPolicyDecision(
        allowed=False,
        requires_approval=False,
        reason=f"Command '{first}' is not in the allowed command policy",
        command_name=first,
    )

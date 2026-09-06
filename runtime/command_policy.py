"""Policy layer for controlled command execution in workspace."""

from __future__ import annotations

import re
import shlex
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
    for op in SHELL_OPERATORS:
        if op in cmd_str:
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


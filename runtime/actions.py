"""Deterministic parsing for model-directed agent actions."""

from __future__ import annotations

import json
import re
from typing import Any


# Allowed read-only workspace tools for Qwen shorthand
WORKSPACE_READONLY_ACTIONS: set[str] = {
    "list_directory",
    "read_file",
    "search_files",
    "find_files",
    "get_file_info",
    "git_status",
    "git_diff",
}

# Mutation tools allowed as Qwen shorthand (require approval at execution)
WORKSPACE_MUTATION_ACTIONS: set[str] = {
    "edit_file",
    "create_file",
    "create_python_script",
    "execute_command",
}

# Tools whose schema permits empty arguments
TOOLS_PERMITTING_EMPTY_ARGS: set[str] = {
    "list_directory",
    "git_status",
    "git_diff",
}

# Required arguments for workspace tools that do not permit empty arguments
TOOL_REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "search_files": ("query",),
    "find_files": ("pattern",),
    "get_file_info": ("path",),
    "edit_file": ("path", "old_text", "new_text"),
    "create_file": ("path", "content"),
    "create_python_script": ("path", "content"),
    "execute_command": ("command",),
}


class ActionParseError(ValueError):
    """Raised when a model response is not a supported structured action."""


def parse_action(raw: str) -> dict[str, Any]:
    """Parse one ``final`` or ``tool`` JSON action from model output.

    Models sometimes wrap JSON in markdown fences or emit a short preamble, so
    a JSON decoder is used to locate the first complete object.  No natural
    language command is executed as a fallback.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
        json_start = cleaned.find("{")
        cleaned = cleaned[json_start:] if json_start >= 0 else ""
    cleaned = re.sub(r"</?(?:tool_call|function_call)>", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        raise ActionParseError("Model returned an empty action.")

    candidates = [cleaned]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))

    decoder = json.JSONDecoder()
    parsed: Any = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            for index, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[index:])
                    break
                except json.JSONDecodeError:
                    continue
            if parsed is not None:
                break

    if not isinstance(parsed, dict):
        raise ActionParseError("Expected a JSON object with action=tool or action=final.")

    action = parsed.get("action")
    if action is not None and not isinstance(action, str):
        raise ActionParseError("Action must be a string.")

    # Qwen-Coder shorthand: {"action": "list_directory", ...} or {"action": "edit_file", ...}
    # Accept exact shorthand when action is an allowlisted workspace tool.
    if isinstance(action, str) and action not in {"tool", "tool_call", "function", "function_call", "final"}:
        if action not in WORKSPACE_READONLY_ACTIONS and action not in WORKSPACE_MUTATION_ACTIONS:
            raise ActionParseError(
                f"Unknown or unauthorized action: '{action}'. "
                f"Allowed actions: 'tool', 'final', or workspace tools."
            )
        tool_args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", parsed.get("input", None))))
        if tool_args is None:
            extra = {k: v for k, v in parsed.items() if k not in {"action", "tool", "name"}}
            if extra:
                tool_args = extra
            elif action in TOOLS_PERMITTING_EMPTY_ARGS:
                tool_args = {}
            else:
                req_args = ", ".join(TOOL_REQUIRED_ARGS.get(action, ("arguments",)))
                raise ActionParseError(f"Tool '{action}' requires arguments: {req_args}.")
        parsed = {"action": "tool", "tool": action, "arguments": tool_args}
        action = "tool"

    if action in ("tool_call", "function", "function_call"):
        action = "tool"
    if action is None and isinstance(parsed.get("tool_call"), dict):
        parsed = parsed["tool_call"]
        action = "tool"
    if action is None and isinstance(parsed.get("function_call"), dict):
        parsed = parsed["function_call"]
        action = "tool"
    if action is None and isinstance(parsed.get("tool_calls"), list) and parsed["tool_calls"]:
        call = parsed["tool_calls"][0]
        if isinstance(call, dict):
            parsed = call.get("function", call)
            action = "tool"
    if action is None and isinstance(parsed.get("name"), str):
        action = "tool"
    if action is None and isinstance(parsed.get("tool"), str):
        action = "tool"
    if action == "final":
        answer = parsed.get("answer")
        if not isinstance(answer, str):
            raise ActionParseError("A final action requires a string answer.")
        return {"action": "final", "answer": answer}

    if action == "tool":
        name = parsed.get("tool") or parsed.get("name")
        arguments = parsed.get(
            "arguments",
            parsed.get("args", parsed.get("parameters", parsed.get("input", None))),
        )
        if arguments is None:
            extra = {
                key: value for key, value in parsed.items()
                if key not in {"action", "tool", "name"}
            }
            if extra:
                arguments = extra
            elif name in TOOLS_PERMITTING_EMPTY_ARGS:
                arguments = {}
            else:
                arguments = {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ActionParseError("Tool arguments must be a JSON object.") from exc
        if not isinstance(name, str) or not name.strip():
            raise ActionParseError("A tool action requires a tool name.")
        name = name.strip()
        if not isinstance(arguments, dict):
            raise ActionParseError("Tool arguments must be a JSON object.")

        # Normalize common argument aliases
        if "path" not in arguments:
            for alt in ("file_path", "filepath", "filename", "file"):
                if alt in arguments and arguments[alt]:
                    arguments["path"] = arguments[alt]
                    break
        if name == "edit_file":
            if "old_text" not in arguments:
                for alt in ("old", "original", "target", "source"):
                    if alt in arguments and arguments[alt]:
                        arguments["old_text"] = arguments[alt]
                        break
            if "new_text" not in arguments:
                for alt in ("new", "replacement", "replace", "new_content"):
                    if alt in arguments and arguments[alt]:
                        arguments["new_text"] = arguments[alt]
                        break
        if name == "execute_command":
            if "command" not in arguments:
                for alt in ("cmd", "run", "exec"):
                    if alt in arguments and arguments[alt]:
                        arguments["command"] = arguments[alt]
                        break

        # Validate required arguments for workspace tools
        if name in TOOL_REQUIRED_ARGS:
            missing = [
                arg for arg in TOOL_REQUIRED_ARGS[name]
                if arg not in arguments or arguments[arg] is None or arguments[arg] == ""
            ]
            if missing:
                raise ActionParseError(
                    f"Tool '{name}' missing required argument(s): {', '.join(missing)}."
                )

        return {"action": "tool", "tool": name, "arguments": arguments}

    if action is None and isinstance(parsed.get("answer"), str):
        return {"action": "final", "answer": parsed["answer"]}

    raise ActionParseError("Unsupported action; expected action=tool or action=final.")

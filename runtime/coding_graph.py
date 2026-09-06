"""Native bounded LangGraph coding loop.

This graph is deliberately small: it reuses AEGIS's existing parser, tool
registry, workspace policy, and approval callbacks. It owns control flow only;
it never grants permissions or implements a second tool backend.
"""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict, Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from runtime.actions import ActionParseError, parse_action
from runtime.task_state import StepStatus
from runtime.verification import verify_tool_result, verify_plan
from runtime.prompts import SYSTEM_PROMPT


def _append(existing: list[Any] | None, new: list[Any] | None) -> list[Any]:
    return [*(existing or []), *(new or [])]


class CodingState(TypedDict, total=False):
    run_id: str
    original_request: str
    normalized_request: str
    status: str
    plan: list[dict[str, Any]]
    current_step_index: int
    tool_call_count: int
    mutation_count: int
    verification: dict[str, Any]
    evidence: list[dict[str, Any]]
    repair_history: list[dict[str, Any]]
    action_signatures: list[str]
    failed_signatures: list[str]
    max_tool_calls: int
    max_mutations: int
    user_request: str
    messages: list[Any]
    allowed_tools: list[str]
    encoded_images: list[str]
    action: dict[str, Any]
    raw_action: str
    tool_result: dict[str, Any]
    tool_results: list[dict[str, Any]]
    events: Annotated[list[dict[str, Any]], _append]
    errors: Annotated[list[str], _append]
    final_answer: str
    phase: str
    iteration: int
    invalid_actions: int
    max_iterations: int
    max_invalid_actions: int
    terminal_status: str


def _event(kind: str, **data: Any) -> dict[str, Any]:
    return {"kind": kind, "timestamp": time.time(), **data}


def build_coding_graph(provider: Any, tool_registry: Any, *, allowed_tools: set[str] | None = None,
                       max_iterations: int = 20, max_invalid_actions: int = 3,
                       on_model_call: Any | None = None, on_tool_call: Any | None = None):
    """Build a compiled reason → validate → execute → record graph."""
    names = set(allowed_tools or tool_registry.list_names())

    async def reason(state: CodingState) -> dict[str, Any]:
        tool_text = "\n".join(f"- {n}: {tool_registry.get(n).description}" for n in sorted(names))
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=(
                SYSTEM_PROMPT + "\n\n"
                "For this coding loop, return exactly one JSON action and no prose. "
                "Never claim a tool ran without its result."
            )), *messages]
        prompt = [*messages, HumanMessage(content=f"Allowed tools:\n{tool_text}\nTask: {state['user_request']}")]
        parts: list[str] = []
        if on_model_call:
            on_model_call()
        async for token in provider.stream_chat(
            [{"role": "system" if isinstance(m, SystemMessage) else "user", "content": str(m.content)} for m in prompt],
            encoded_images=state.get("encoded_images", []),
        ):
            parts.append(token)
        raw = "".join(parts)
        next_iteration = state.get("iteration", 0) + 1
        events = [_event("reason", iteration=next_iteration)]
        if state.get("iteration", 0) == 0:
            events = [_event("plan_created", goal=state["user_request"][:500]),
                      _event("plan_validated", status="bounded_tool_loop"), *events]
        return {"raw_action": raw, "phase": "validate", "iteration": state.get("iteration", 0) + 1,
                "status": "running", "events": events}

    async def validate_action(state: CodingState) -> dict[str, Any]:
        if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 40):
            return {"phase": "safe_error", "errors": ["tool_call_budget_exhausted"],
                    "events": [_event("safe_error", error="tool_call_budget_exhausted")]}
        try:
            action = parse_action(state.get("raw_action", ""))
        except ActionParseError as exc:
            # Some local providers switch to concise prose after a successful
            # tool call.  Treat that prose as the final answer, but only after
            # real evidence exists; before the first tool it remains invalid.
            raw = state.get("raw_action", "").strip()
            if state.get("tool_results") and raw:
                return {"final_answer": raw, "terminal_status": "success", "phase": "finish"}
            count = state.get("invalid_actions", 0) + 1
            return {"invalid_actions": count, "phase": "reason" if count < state.get("max_invalid_actions", 3) else "safe_error",
                    "errors": [str(exc)], "events": [_event("invalid_action", error=str(exc)[:300], count=count)]}
        if action["action"] == "final":
            return {"action": action, "final_answer": action["answer"], "terminal_status": "success", "phase": "finish"}
        if action["tool"] not in names:
            return {"phase": "safe_error", "errors": [f"unauthorized tool: {action['tool']}"],
                    "events": [_event("blocked", tool=action["tool"])]}
        if action["tool"] in {"edit_file", "create_file", "create_python_script", "create_checkpoint", "restore_checkpoint"} and state.get("mutation_count", 0) >= state.get("max_mutations", 10):
            return {"phase": "safe_error", "errors": ["mutation_budget_exhausted"],
                    "events": [_event("blocked", tool=action["tool"], error="mutation_budget_exhausted")]}
        signature = json.dumps({"tool": action["tool"], "arguments": action.get("arguments", {})},
                               sort_keys=True, default=str)
        signatures = [*state.get("action_signatures", []), signature]
        return {"action": action, "phase": "execute_tool", "events": [_event(
            "tool_requested", tool=action["tool"], arguments=action.get("arguments", {}),
            iteration=state.get("iteration", 0))], "action_signatures": signatures}

    async def execute_tool(state: CodingState) -> dict[str, Any]:
        action = state["action"]
        name, args = action["tool"], action.get("arguments", {})
        started = time.perf_counter()
        if on_tool_call:
            on_tool_call()
        try:
            result = await tool_registry.get(name).ainvoke(args)
            if not isinstance(result, dict):
                result = {"ok": True, "result": str(result)}
        except Exception as exc:
            result = {"ok": False, "tool": name, "error": type(exc).__name__, "message": str(exc)[:300]}
        result = dict(result)
        result.setdefault("tool", name)
        result.setdefault("duration_ms", round((time.perf_counter() - started) * 1000, 2))
        result_text = json.dumps(result, ensure_ascii=False, default=str)
        if name == "edit_file":
            trace_args = {"path": str(args.get("path", ""))[:200],
                          "old_text_length": len(str(args.get("old_text", ""))),
                          "new_text_length": len(str(args.get("new_text", "")))}
        elif name in {"create_file", "create_python_script"}:
            trace_args = {"path": str(args.get("path", ""))[:200],
                          "content_length": len(str(args.get("content", args.get("source", ""))))}
        else:
            trace_args = {k: str(v)[:200] for k, v in args.items()}
        approval = None
        if name in {"edit_file", "create_file", "create_python_script"}:
            approval = "approved" if result.get("ok", True) else ("denied" if result.get("error") == "approval_denied" else None)
        events = [_event(
                "tool_result", tool=name, arguments=trace_args,
                result=result_text[:12000], result_size=len(result_text.encode("utf-8")),
                truncated=len(result_text.encode("utf-8")) > 12000,
                status="success" if result.get("ok", True) else "failure", duration_ms=result["duration_ms"],
                iteration=state.get("iteration", 0),
                approval=(approval or result.get("approval")))
        ]
        # Mutation tools are immediately read back through the same registry.
        # This makes file-generation/edit claims evidence-based without asking
        # the model to remember a second verification step.
        if name in {"edit_file", "create_file", "create_python_script"} and result.get("ok", True):
            try:
                read_tool = tool_registry.get("read_file")
            except Exception:
                read_tool = None
            if read_tool is not None and args.get("path"):
                verify_started = time.perf_counter()
                try:
                    verify = await read_tool.ainvoke({"path": args["path"]})
                    verify_ok = isinstance(verify, dict) and verify.get("ok", False)
                    verify_payload = verify if isinstance(verify, dict) else {"result": str(verify)}
                    verify_content = str(verify_payload.get("content", ""))
                    expected = str(args.get("new_text", ""))
                    if name == "edit_file" and verify_ok:
                        if expected and expected in verify_content:
                            verify_payload = {"ok": True, "message": f"edit confirmed — new_text present in {args['path']}"}
                        else:
                            verify_ok = False
                            verify_payload = {"ok": False, "message": f"edit verification failed — new_text not found in {args['path']}"}
                    elif not verify_ok:
                        verify_payload = {"ok": False, "message": f"failed to read {args['path']}: {verify_payload.get('error', 'unknown error')}"}
                    else:
                        verify_payload = {"ok": True, "path": str(args["path"]),
                                          "content_length": len(verify_content),
                                          "message": f"file exists and is readable: {args['path']}"}
                except Exception as exc:
                    verify_ok = False
                    verify_payload = {"ok": False, "error": type(exc).__name__, "message": str(exc)[:300]}
                verify_text = json.dumps(verify_payload, ensure_ascii=False, default=str)
                events.append(_event(
                    "tool_result", tool="read_file", verification=True,
                    arguments={"path": str(args["path"])[:200]},
                    result=verify_text[:12000], result_size=len(verify_text.encode("utf-8")),
                    truncated=len(verify_text.encode("utf-8")) > 12000,
                    status="success" if verify_ok else "failure",
                    duration_ms=0,
                    iteration=state.get("iteration", 0)))
                result["post_edit_verification"] = {"tool": "read_file", "ok": verify_ok}
        ordinal = int(state.get("tool_call_count", 0)) + 1
        step = {"step_id": f"step-{ordinal}", "ordinal": ordinal,
                "description": f"Execute {name}", "tool_name": name, "parameters": trace_args,
                "expected_evidence": [{"kind": "tool_result", "passed": True}],
                "status": StepStatus.SUCCEEDED.value if result.get("ok", True) else StepStatus.FAILED.value,
                "result": result, "retry_count": 0, "max_retries": 2}
        return {"tool_result": result, "phase": "record_result", "plan": [*state.get("plan", []), step],
                "tool_call_count": ordinal,
                "mutation_count": state.get("mutation_count", 0) + (1 if name in {"edit_file", "create_file", "create_python_script"} else 0),
                "events": events}

    async def record_result(state: CodingState) -> dict[str, Any]:
        result = state.get("tool_result", {})
        results = [*state.get("tool_results", []), result]
        tool_name = result.get("tool", "unknown")
        messages = [*state.get("messages", []), HumanMessage(content=f"TOOL RESULT [{tool_name}]: {json.dumps(result, default=str)[:12000]}")]
        verification = verify_tool_result(result)
        step_events = [_event("step_succeeded" if verification.get("passed") else "step_failed",
                              tool=tool_name, verification=verification)]
        if not result.get("ok", True):
            # Return control to the model so it can diagnose and choose the
            # next safe action; the error remains structured evidence.
            failed_signature = json.dumps({"tool": tool_name, "arguments": state.get("action", {}).get("arguments", {})}, sort_keys=True, default=str)
            failed_signatures = [*state.get("failed_signatures", []), failed_signature]
            if failed_signatures.count(failed_signature) >= 3:
                return {"tool_results": results, "messages": messages, "phase": "safe_error",
                        "status": "blocked", "errors": ["no_progress_detected: repeated identical tool failure"],
                        "failed_signatures": failed_signatures,
                        "verification": verification, "evidence": verification.get("evidence", []),
                        "events": [*step_events, _event("no_progress_detected", tool=tool_name)]}
            return {"tool_results": results, "messages": messages, "phase": "self_heal",
                    "status": "healing", "verification": verification,
                    "failed_signatures": failed_signatures,
                    "evidence": verification.get("evidence", []),
                    "events": [*step_events, _event("repair_requested", tool=tool_name)]}
        return {"tool_results": results, "messages": messages, "phase": "verify_plan",
                "status": "running", "verification": verification,
                "evidence": verification.get("evidence", []), "events": step_events}

    async def verify_plan_node(state: CodingState) -> dict[str, Any]:
        """Deterministically verify the latest tool result before continuing."""
        check = verify_plan((state.get("tool_results") or [])[-1:])
        if not check.get("passed"):
            return {"verification": check, "phase": "self_heal", "status": "healing",
                    "events": [_event("verification_failed", summary=check.get("summary"))]}
        return {"verification": check, "phase": "reason", "status": "running",
                "events": [_event("verification_passed", summary=check.get("summary"))]}

    async def self_heal(state: CodingState) -> dict[str, Any]:
        """Ask the same bounded local reasoner for the next corrected action."""
        return {**(await reason(state)), "status": "healing",
                "events": [_event("self_heal", retry_count=state.get("invalid_actions", 0))]}

    def route(state: CodingState) -> str:
        if state.get("phase") in {"reason", "self_heal", "verify_plan"}:
            if state.get("phase") == "verify_plan":
                return "verify_plan"
            if state.get("iteration", 0) >= state.get("max_iterations", 20):
                return "safe_error"
            return state.get("phase", "reason")
        return state.get("phase", "safe_error")

    def finish(state: CodingState) -> dict[str, Any]:
        return {"terminal_status": state.get("terminal_status", "success"), "phase": "finish",
                "events": [_event("finish", status=state.get("terminal_status", "success"))]}

    def safe_error(state: CodingState) -> dict[str, Any]:
        errors = state.get("errors", []) or ["coding loop stopped safely"]
        if state.get("iteration", 0) >= state.get("max_iterations", 20):
            message = f"Maximum tool-loop iterations ({state.get('max_iterations', 20)}) reached."
        elif state.get("invalid_actions", 0) >= state.get("max_invalid_actions", 3):
            message = "The model produced too many invalid tool actions."
        else:
            message = f"Unable to complete safely: {errors[-1]}"
        return {"terminal_status": "blocked", "final_answer": message,
                "phase": "finish", "events": [_event("safe_error", error=errors[-1])]} 

    graph = StateGraph(CodingState)
    graph.add_node("reason", reason)
    graph.add_node("validate_action", validate_action)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("record_result", record_result)
    graph.add_node("verify_plan", verify_plan_node)
    graph.add_node("self_heal", self_heal)
    graph.add_node("finish", finish)
    graph.add_node("safe_error", safe_error)
    graph.add_edge(START, "reason")
    graph.add_edge("reason", "validate_action")
    graph.add_conditional_edges("validate_action", route,
                                {"reason": "reason", "execute_tool": "execute_tool",
                                 "self_heal": "self_heal", "finish": "finish", "safe_error": "safe_error"})
    graph.add_edge("execute_tool", "record_result")
    graph.add_conditional_edges("record_result", route, {"verify_plan": "verify_plan", "reason": "reason", "self_heal": "self_heal", "safe_error": "safe_error"})
    graph.add_conditional_edges("verify_plan", route, {"reason": "reason", "self_heal": "self_heal", "safe_error": "safe_error"})
    graph.add_edge("self_heal", "validate_action")
    graph.add_edge("finish", END)
    graph.add_edge("safe_error", END)
    return graph.compile()


async def run_coding_graph(provider: Any, tool_registry: Any, request: str, *, allowed_tools: set[str] | None = None,
                           messages: list[Any] | None = None,
                           encoded_images: list[str] | None = None,
                           max_iterations: int = 20, max_invalid_actions: int = 3,
                           on_model_call: Any | None = None, on_tool_call: Any | None = None,
                           max_tool_calls: int = 40, max_mutations: int = 10) -> CodingState:
    graph = build_coding_graph(provider, tool_registry, allowed_tools=allowed_tools,
                               max_iterations=max_iterations, max_invalid_actions=max_invalid_actions,
                               on_model_call=on_model_call, on_tool_call=on_tool_call)
    return await graph.ainvoke({"user_request": request, "original_request": request,
                                "normalized_request": request, "status": "planned",
                                "messages": messages or [HumanMessage(content=request)],
                                "allowed_tools": list(allowed_tools or tool_registry.list_names()),
                                "encoded_images": encoded_images or [], "plan": [], "current_step_index": 0,
                                "tool_call_count": 0, "mutation_count": 0, "verification": {}, "evidence": [],
                                "repair_history": [],
                                "action_signatures": [],
                                "failed_signatures": [],
                                "max_tool_calls": max_tool_calls, "max_mutations": max_mutations,
                                "events": [], "tool_results": [], "errors": [], "iteration": 0,
                                "invalid_actions": 0, "max_iterations": max_iterations,
                                "max_invalid_actions": max_invalid_actions, "phase": "reason"})

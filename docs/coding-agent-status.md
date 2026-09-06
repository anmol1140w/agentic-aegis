# AEGIS coding-agent execution status

The production terminal/API path is a bounded universal LangGraph `StateGraph`
entrypoint (`Orchestrator.run_master`). Coding specialist tool calls then use
the native coding subgraph, which runs the complete
`reason → validate_action → execute_tool → record_result` loop. The small
`Orchestrator._stream_tool_loop` method is only an event-shape adapter for the
existing CLI stream; it does not contain a second execution loop. Each action
is parsed as an allowlisted JSON tool call, validated by policy, executed
through the real workspace tool, and returned to the model as a bounded result
before the next decision.

Coding actions currently supported include workspace listing/search/read,
Git status/diff, approved file creation/editing (including syntax validation
for Python scripts), and policy-controlled command execution. Edits require
read-before-edit and terminal approval. Test commands return stdout, stderr,
exit code, duration, timeout, truncation, and approval metadata. Failed
commands remain failures; the model cannot fabricate a passing result.

Before each command, the local process-group sandbox performs a readiness
probe. During execution the CLI reports the sandbox boundary, command state,
bounded stdout/stderr lines, exit code, and final stopped state. Commands run
with a workspace-rooted cwd, sanitized environment, policy validation, output
limits, and process-group cleanup on timeout.

The loop is bounded by the configured maximum iterations and invalid-action
limit. It rejects malformed actions, unknown tools, path escapes, shell
operators, denied binaries, and unsafe Python expressions. Successful
mutations trigger an immediate bounded `read_file` verification before the
next model decision. The Master path records concise operational events and
persists run traces without private chain-of-thought.

The legacy `astream`/`ainvoke` classifier graph remains only as a compatibility
surface for older library callers. Normal CLI/API requests do not enter it;
they use the universal graph and the same registry, policy, approval, and tool
implementations.

## Plan, evidence, and bounded repair

`runtime/task_state.py` defines serializable task and step status contracts.
`runtime/plan_parser.py` accepts only bounded JSON plans and rejects unknown
tools before execution. `runtime/verification.py` evaluates command/file
evidence deterministically, while `runtime/self_healing.py` classifies whether
a failure is safe to retry and validates repair-action shapes. The native graph
records `plan_created`, `plan_validated`, `step_succeeded`, `step_failed`,
`repair_requested`, and verification evidence in state. Model prose can explain
an outcome, but cannot turn failed deterministic evidence into success.

This follows the useful high-level plan/execute/retry behavior observed in the
public `ollama-desktop-agent` reference while keeping AEGIS's LangGraph,
workspace policy, approvals, and structured audit boundaries; no monolithic
controller or reference code is copied.

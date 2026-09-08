# AEGIS execution architecture

## Production rebuild principles

The architecture follows the supplied production-agentic references and uses
the least-autonomous execution mode that can satisfy a request:

| Mode | Use when | Required controls |
|---|---|---|
| Deterministic | exact greetings, calculations, health/status, or fixed transformations | typed result and deterministic verification |
| Workflow | the task has a stable sequence such as extraction, OCR, or validation | explicit state, bounded edges, replayable evidence |
| Agent | the task needs bounded adaptation and tool choice | allowlisted tools, deadlines, structured actions |
| Hybrid | the task is multi-step, context-heavy, or high-impact | planner/executor, checkpoints, independent review, approval |

`aegis.architecture.ArchitecturePolicy` records this decision in every task
specification. The model cannot promote a deterministic or high-risk task into
unbounded autonomy.

The target production topology is:

```text
request → architecture policy → planner → typed plan validation
        → executor/specialist → tool gateway → sandbox or MCP adapter
        → observation/event stream → critic/verification → approval/deliverable
        → episodic trace + evaluation record
```

This is intentionally a hybrid of the Planner-Executor and Supervisor
patterns. A single agent is reserved for tasks where adaptive tool choice is
actually necessary; predictable work remains a workflow.

AEGIS uses a bounded local execution protocol:

```text
request → repository index → bounded context → plan → validate → execute
        → observe → deterministic verification → independent review → evidence
```

The small local model proposes narrowly scoped actions. Python owns state,
workspace policy, tool contracts, retries, checkpoints, and completion evidence.
The SQLite repository index extracts file metadata, Python/common-language
symbols, imports, and test candidates. `aegis.retrieval.HybridKnowledgeStore`
provides a separate offline knowledge store with lexical ranking and optional
embedding similarity. `runtime/model_profiles.py`
defines the model-agnostic CPU-first 5B operating limits and keeps the existing
the configured coder model; model tags are owned by `config/models.yaml`.

Security and policy failures are non-retryable. Recoverable failures are
normalized into the failure taxonomy and may use at most the configured repair
budget. The deterministic reviewer checks changed-file evidence and verification
before a task can complete.

Offline benchmark scoring is available through `benchmarks/runner.py`; it can
compare `coder_only` and `aegis` runners without contacting a model provider.

The SIH presentation workflow uses the supplied six-slide template and is
documented in `docs/aegis-platform.md` and the repository README.

## Memory and evaluation contract

AEGIS memory is layered: short-term context belongs to the current run,
episodic memory stores replayable events, semantic memory stores retrievable
facts, and relational memory is reserved for structured links. Retention,
scope, provenance, importance, and deletion are explicit fields.

Agent evaluation records correctness, latency, recovery, hallucination, and
tool errors. Trajectory evaluation additionally measures tool precision, tool
recall, tool efficiency, retries, policy violations, token usage, and
completion. A final answer is never sufficient evidence by itself.

# AEGIS architecture — authoritative current version

AEGIS has one production execution boundary: `Orchestrator.run_master`.
The CLI and FastAPI API both enter the same universal task graph. The legacy
classifier graph remains only for compatibility with older library callers.

```text
CLI/API
  → NLP normalization
  → MasterAgent plan and capability routing
  → AgentRegistry specialist
  → typed ToolRegistry + path/command/approval policy
  → workspace tool or local sandbox
  → observation and read-back
  → deterministic verification
  → independent review
  → bounded repair or safe failure
  → result, trace, network report, evidence
```

## Runtime ownership

| Layer | Owns |
|---|---|
| MasterAgent | request interpretation, plan, specialist selection, delegation, review |
| AgentRegistry | specialist roles, model aliases, allowed tools |
| ToolRegistry/runtime | schemas, risk, approval, timeout, audit identity |
| WorkspaceReadTools | canonical `./workspace` paths, file changes, commands, checkpoints |
| RepositoryIndex | SQLite metadata, symbols, imports, tests, config, relevance ranking |
| Verification/reviewer | exit codes, changed-file evidence, read-back, diff and policy checks |
| NetworkMonitor | live sockets plus named local/external model/tool counters |

Every action follows `decide → validate → execute → observe → update → verify`.
Mutations are serialized, checkpoint-protected, approval-gated, and never
performed concurrently.

The `aegis.sovereign.SovereignExecutor` typed-plan seam is not a second normal
task runner. It exists for explicit workflow integrations and contract tests;
normal CLI/API requests remain on `Orchestrator.run_master` and
`runtime.task_graph`. Optional MCP, RL, A2A, TEE, federation, Kubernetes, and
ledger adapters are not required on that path. The implemented `aegis/`
contracts are available to API consumers and can be adopted by future
executors without changing the normal CLI route.

## Platform contracts

`aegis.contracts.ExecutionPlan` represents a typed dependency graph.
`PlanValidator` checks dependencies, cycles, allowed tools, allowed models, and
capability compatibility before execution. `WorkflowRegistry` stores named
versioned plans.

`aegis.governance` provides role/group permissions, department model/tool
policies, and `AuditChain`, an append-only hash-chained JSONL log. Retrieval is
provided by `HybridKnowledgeStore`, which works offline with lexical ranking
and can accept an optional embedding callback. `MemoryStore` provides scoped
SQLite memory with TTL. `BackgroundJobManager` provides bounded retries and
cancellation.

## Agents

- `master_agent`: only production orchestrator.
- `coding_agent`: source inspection, edits, targeted tests, bounded repair.
- `document_agent`: local PDF/DOCX/PPTX/TXT workflows and artifact verification.
- `vision_agent`: local image preprocessing and Ollama vision analysis.
- `general_agent`: non-specialized local reasoning.

The former lightweight route is not an initial handoff. The master routes
directly to a real specialist and sends the result back through verification.

## Boundaries

The canonical user workspace is `./workspace`; no tool may resolve outside it.
The default command backend is a sanitized process group. Docker and Linux
Bubblewrap are optional adapters. C++17 process lifecycle code in `native/` is
an optional helper; Python remains the portable fallback.

# AEGIS platform layer

The repository now exposes a provider-neutral platform layer under `aegis/`.
It is designed to let the existing LangGraph-style runtime, Ollama models,
MCP adapters and sandbox tools share one set of contracts.

## Implemented contracts

- `aegis.contracts`: typed execution plans, dependency-cycle detection,
  capability/model/tool compatibility checks, and versioned workflow registry.
- `aegis.governance`: role and group permissions, department-level model/tool
  policy, and tamper-evident hash-chained audit events.
- `aegis.retrieval`: offline hybrid lexical retrieval with optional embedding
  similarity and source/metadata citations.
- `aegis.memory`: scoped SQLite memory with TTL retention and deletion.
- `aegis.evaluation`: persistent records for correctness, latency, tool errors,
  recovery, and hallucination measurements.
- `runtime.background`: bounded, retryable, cancellable background jobs.

## API surfaces

The FastAPI boundary includes workflow validation/registration, knowledge
ingestion/search, memory, background jobs, audit verification and evaluation
summary endpoints. Authentication can replace the local principal adapter
without changing policy checks.

## MVP execution core

The executable, demonstrable path is deliberately small:

```text
RBAC → policy → typed plan → preflight validation → capability router
     → supervisor → local RAG / MCP / container sandbox
     → deterministic verification → human gate (when risk requires it)
     → deliverable → tamper-evident audit chain
```

`aegis.sovereign.SovereignExecutor` enforces the workflow portion of that path: **validate →
deterministic route → risk approval → execute → verify → audit**. A failed
required step ends the run. Retriable steps emit a checkpoint before retrying.
No model response by itself is accepted as completion.

Set `require_citations` or `require_deliverable` on a plan step to make those
acceptance gates mandatory. Citation receipts must identify a source or URL;
deliverable receipts must point to a file that exists at verification time.

## Optional extension APIs

Advanced capabilities are interfaces, not required pipeline stages. They are
only selected by an explicit plan/backend configuration:

| Extension interface | Current implementation status | Honest claim |
|---|---|---|
| `RoutingStrategy` | Deterministic local router is default; `RLRoutingStrategy` is opt-in | Learned routing is experimental and falls back in shadow/disabled mode |
| `OptimizationStrategy` / `DSPyProgram` | Program contract | DSPy-compatible optimization boundary |
| `AgentTransport` / `A2AGateway` | In-process transport | A2A extension point, not a mandatory hop |
| `InferenceBackend` / `FederatedInference` | Local-worker fan-out contract | Federated inference extension; not used by the MVP demo |
| `ExecutionIsolationBackend` / `TEEVerifier` | Measurement verifier | TEE verification interface; MVP execution is container-isolated, not TEE-secured |
| `DeploymentBackend` / `KubernetesPlanner` | Manifest planning | Kubernetes deployment planning, not live cluster deployment |
| `AuditAnchorBackend` / `BlockchainAuditLedger` | Experimental in-memory adapter | Tamper-evident local audit chain; no blockchain anchoring claim |

The local audit chain remains complete while air-gapped. An external ledger,
when implemented, can anchor its current hash through `AuditAnchorBackend` but
must never be required for auditing to work.

`VisualWorkflowEditor.export` produces canvas node/edge JSON while
`ExecutionPlan` remains the single source of execution truth.

## PS demonstration scenarios

1. **Cited document triage:** retrieve local sources, create a deliverable, and
   show that missing citations or a missing artifact fail verification.
2. **Safe coding repair:** run a coding task in the local sandbox, record the
   test result, and demonstrate a checkpoint/retry recovery.
3. **High-risk request:** request a destructive or production-impacting tool;
   show that policy creates a human gate and the tool never runs without approval.

For a local demo, register a plan with `POST /api/workflows`, inspect its canvas
at `/api/workflows/{name}/{version}/visual`, then call
`POST /api/workflows/execute`. The API uses deterministic routing by default;
high-risk steps require `approve_high_risk=true`. Kubernetes manifest planning
is available through `/api/kubernetes/manifest`; applying a manifest remains an
explicit external deployment action.

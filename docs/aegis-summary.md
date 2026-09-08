# AEGIS current system summary

AEGIS is a local-first, Master-first workbench for coding, documents, vision,
retrieval, and verified artifacts. It uses local Ollama inference and does not
use cloud fallbacks in the normal path. The platform layer also exposes typed
plans, governance, memory, evaluation, background jobs, and auditable events.

## Current flow

```text
request → NLP → Master plan → specialist → typed tools
        → policy/approval → observation → verification → review → evidence
```

The final response is derived from structured state, not from an unverified
model claim. A successful mutation must have changed-file evidence, read-back,
syntax/static checks where applicable, successful command evidence when
requested, a reviewer result, and a checkpoint or explicit safe path.

## Local roles and model aliases

| Role | Config alias | Default model |
|---|---|---|
| Master/general/document | `qwen-general` | configured in `config/models.yaml` |
| Coding | `qwen-coder` | configured in `config/models.yaml` |
| Vision | `qwen-vision` | configured in `config/models.yaml` |
| Lightweight capability profile | `llama-small` | configured locally; not an initial handoff |

Change model tags only in `config/models.yaml`.

## Workspace products

`workspace/fixtures` contains reusable offline examples, `workspace/agent_test`
contains agent contract tests, `workspace/artifacts` stores verified reusable
files with provenance, and `workspace/executions` stores bounded command
records. Per-run result and trace files are generated under
`workspace/outputs/<run_id>/`.

Platform runtime state lives in `.aegis/` and `logs/aegis-chain.jsonl`.
It includes checkpoints, scoped memory, evaluation records, and a hash-chained
audit trail. These files are local operational state, not source artifacts.

## Current platform capabilities

| Capability | Current implementation |
|---|---|
| Typed workflows | `aegis.contracts.ExecutionPlan`, validation, registry |
| Governance | RBAC, group permissions, department model/tool policy |
| Retrieval | Offline hybrid lexical retrieval with optional embeddings |
| Memory | Scoped SQLite records with TTL and deletion |
| Long-running work | Bounded retryable and cancellable background jobs |
| Evaluation | Persistent quality, latency, recovery, and error metrics |
| Audit | Append-only JSONL plus hash-chain verification |

## Explicit boundaries

Cloud inference, unrestricted shell, network tools, Git mutation, delete tools,
and desktop control remain restricted or disabled by default. Embedding
retrieval is optional and only activates when an embedding callback is
configured. A2A, RL routing, TEE, federation, Kubernetes, blockchain anchoring,
and live identity federation remain extension points, not default runtime
dependencies. Silent completion claims are never accepted.

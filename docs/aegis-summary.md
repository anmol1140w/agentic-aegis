# AEGIS current system summary

AEGIS is a local-first, Master-first workbench for coding, documents, vision,
and verified artifacts. It uses local Ollama inference and does not use cloud
fallbacks in the normal path.

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
| Master/general/document | `qwen-general` | `qwen3.5:9b` |
| Coding | `qwen-coder` | `qwen2.5-coder:7b` |
| Vision | `qwen-vision` | `qwen3-vl:8b` |
| Lightweight capability profile | `llama-small` | configured locally; not an initial handoff |

Change model tags only in `config/models.yaml`.

## Workspace products

`workspace/fixtures` contains reusable offline examples, `workspace/agent_test`
contains agent contract tests, `workspace/artifacts` stores verified reusable
files with provenance, and `workspace/executions` stores bounded command
records. Per-run result and trace files are generated under
`workspace/outputs/<run_id>/`.

## Explicit non-goals

Cloud inference, unrestricted shell, network tools, Git mutation, delete tools,
desktop control, embeddings/vector retrieval, and silent completion claims are
not enabled by the current architecture.

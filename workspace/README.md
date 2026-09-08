# AEGIS Workspace

This directory is the controlled local workspace for AEGIS tasks. Agents may
read and search files here; writes and approved commands remain policy-gated.

Subdirectories:

- `fixtures/` — stable demo and test inputs.
- `artifacts/` — registered generated artifacts.
- `executions/` — execution state and metadata.
- `outputs/` — per-run results and traces (generated at runtime).
- `temporary/` — short-lived processing data (cleaned after use).

Repository-level platform state lives outside this controlled content area in
`.aegis/` and `logs/`. It includes checkpoints, scoped memory, evaluations,
and the hash-chained audit log.

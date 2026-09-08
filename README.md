# AEGIS — Sovereign Agent Workbench

AEGIS is a local-first coding, document, vision, retrieval, and verification
workbench. The Master agent owns routing and state; specialists operate through
typed, workspace-confined tools. The platform layer adds typed workflow plans,
preflight validation, RBAC, policy enforcement, durable memory, hybrid local
retrieval, background jobs, evaluation records, and hash-chained audit events.
Completion requires read-back, command evidence, verification, and review.
Normal execution does not use cloud APIs.

## Quick start

Requirements: Git, Python 3.10–3.13, and Ollama for model-backed tasks. The
CLI works without Ollama for deterministic tools and reports unavailable models
instead of silently using a cloud service.

To change model tags or local dependency names in one place, copy
`.env.example` to `.env`. Shell environment variables take precedence; the
logical model aliases and capability routing remain defined by AEGIS.

```bash
git clone <repository-url> aegis
cd aegis
bash scripts/setup.sh
source .venv/bin/activate
# Pull the model tags listed in config/models.yaml with Ollama.
python cli.py
```

Windows PowerShell:

```powershell
git clone <repository-url> aegis
cd aegis
.\scripts\setup.ps1
# Pull the model tags listed in config/models.yaml with Ollama.
python cli.py
```

macOS uses the same `scripts/setup.sh` flow as Linux. The setup scripts install
the project from `pyproject.toml`; it is the single dependency source. OCR is
optional because PaddleOCR wheel availability differs by operating system.

The supplied SIH2026 deck is also retained as the personal
`artifact-template-sih2026-aegis-presentation` template. Use it for future
six-slide SIH project presentations and replace the bracketed portal fields
before submission.

## Architecture

```text
CLI / API → NLP normalization → MasterAgent plan and routing
  → AgentRegistry specialist → typed tool policy
  → workspace mutation or sandboxed command → read-back
  → deterministic verification → independent review → evidence

```

Platform services add this contract-driven path:

```text
Principal → RBAC/policy → typed ExecutionPlan → preflight validation
  → local router/supervisor → RAG/MCP/sandbox → verification
  → approval when required → deliverable → hash-chained audit
```

The active path is `Orchestrator.run_master`. The older classifier graph is
kept only for library compatibility; it is not the normal CLI/API route.

| Role | Default model | Responsibility |
|---|---|---|
| Master | configured `qwen-general` tag | intent, plan, delegation, review, recovery |
| Coding | configured `qwen-coder` tag | inspect, edit, run targeted checks |
| Vision | configured `qwen-vision` tag | local image analysis |
| Document/general | configured `qwen-general` tag | document workflows and general reasoning |
| Lightweight profile | configured capability only | compatibility profile; not an initial handoff |

## Workspace and safety

The canonical user workspace is `./workspace`. Relative paths resolve there;
absolute paths, traversal, symlink escapes, network commands, deletes, Git
mutation, and arbitrary shell are denied or approval-gated. Run evidence is
written under `workspace/outputs/`, artifacts under `workspace/artifacts/`, and
command records under `workspace/executions/`.

Platform runtime state is stored under `.aegis/` and `logs/`: checkpoints,
scoped SQLite memory, evaluation records, and the platform audit chain. These
are local runtime files and should not be committed.

## API platform endpoints

In addition to `/api/tasks`, the local API exposes workflow validation and
registration, knowledge ingestion/search, scoped memory, retryable background
jobs, audit verification, and evaluation summaries. See
[docs/aegis-api.md](docs/aegis-api.md) for request examples.

## Commands

```text
/models    show local model availability
/network   show connection and model/tool counters
/sandbox   check the active command sandbox
/status    show operating policy
/quit      exit
```

Examples: `create a Python file about linked lists and run its tests`,
`inspect the repository tree`, and `analyze workspace/fixtures/sample_task.json`.

## Testing

```bash
.venv/bin/pytest -q                       # Linux/macOS
.venv/Scripts/python.exe -m pytest -q     # Windows
```

The suite is offline by default. Live Ollama tests are opt-in. Run
`git diff --check` before sharing changes.

## Documentation

- [SETUP.md](SETUP.md) — exact Linux, macOS, Windows, Ollama, sandbox, and API setup.
- [docs/architecture.md](docs/architecture.md) — authoritative architecture.
- [docs/workflows.md](docs/workflows.md) — Master and specialist lifecycle.
- [docs/tools.md](docs/tools.md) — tool families and safety contracts.
- [docs/models.md](docs/models.md) — model aliases and local routing.
- [docs/security.md](docs/security.md) — boundaries and approvals.
- [docs/aegis-api.md](docs/aegis-api.md) — local FastAPI interface.
- [docs/aegis-platform.md](docs/aegis-platform.md) — typed plans, governance, retrieval, memory, jobs, and evaluation.
- [docs/native-boundary.md](docs/native-boundary.md) — optional C++ process helper.

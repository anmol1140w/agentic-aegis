# AEGIS — Sovereign Agent Workbench

AEGIS is a local-first coding, document, vision, and verification workbench.
The Master agent owns routing and state; specialists operate through typed,
workspace-confined tools. Completion requires read-back, command evidence,
verification, and review. Normal execution does not use cloud APIs.

## Quick start

Requirements: Git, Python 3.10–3.13, and Ollama for model-backed tasks. The
CLI works without Ollama for deterministic tools and reports unavailable models
instead of silently using a cloud service.

```bash
git clone <repository-url> aegis
cd aegis
bash scripts/setup.sh
source .venv/bin/activate
ollama pull qwen3.5:9b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-vl:8b
python cli.py
```

Windows PowerShell:

```powershell
git clone <repository-url> aegis
cd aegis
.\scripts\setup.ps1
ollama pull qwen3.5:9b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-vl:8b
python cli.py
```

macOS uses the same `scripts/setup.sh` flow as Linux. The setup scripts install
the project from `pyproject.toml`; it is the single dependency source. OCR is
optional because PaddleOCR wheel availability differs by operating system.

## Architecture

```text
CLI / API → NLP normalization → MasterAgent plan and routing
  → AgentRegistry specialist → typed tool policy
  → workspace mutation or sandboxed command → read-back
  → deterministic verification → independent review → evidence
```

The active path is `Orchestrator.run_master`. The older classifier graph is
kept only for library compatibility; it is not the normal CLI/API route.

| Role | Default model | Responsibility |
|---|---|---|
| Master | `qwen3.5:9b` | intent, plan, delegation, review, recovery |
| Coding | `qwen2.5-coder:7b` | inspect, edit, run targeted checks |
| Vision | `qwen3-vl:8b` | local image analysis |
| Document/general | `qwen3.5:9b` | document workflows and general reasoning |
| Lightweight profile | configured capability only | compatibility profile; not an initial handoff |

## Workspace and safety

The canonical user workspace is `./workspace`. Relative paths resolve there;
absolute paths, traversal, symlink escapes, network commands, deletes, Git
mutation, and arbitrary shell are denied or approval-gated. Run evidence is
written under `workspace/outputs/`, artifacts under `workspace/artifacts/`, and
command records under `workspace/executions/`.

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
- [docs/native-boundary.md](docs/native-boundary.md) — optional C++ process helper.

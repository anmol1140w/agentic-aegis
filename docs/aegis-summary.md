# AEGIS — complete system summary

AEGIS is a local-first, Master-first agent workbench for secure coding,
document, vision, and artifact workflows. It runs through local Ollama models,
keeps workspace data local, and requires deterministic evidence before it
reports success.

## Runtime flow

```text
User / CLI / API
  → NLP preprocessing
  → MasterAgent
  → capability registry and semantic matching
  → universal LangGraph task graph
  → specialist agent
  → policy / approval
  → bounded tools and local pipelines
  → structured AgentResult
  → verify_plan
  → review, bounded repair, or safe failure
  → final response and audit output
```

The universal graph nodes are `normalize_request`,
`classify_and_route`, `create_plan`, `validate_plan`, `execute_step`,
`record_step_result`, `verify_plan`, `self_heal`, `review_and_finish`, and
`safe_failure`. Coding has an additional bounded tool loop for inspect → edit
→ test → readback → repair behavior.

## Agents and models

| Agent | Local model | Responsibility |
|---|---|---|
| `master_agent` | `qwen3.5:9b` | intent, capability selection, delegation, review, repair |
| `coding_agent` | `qwen2.5-coder:7b` | source inspection, approved edits, commands, tests |
| `document_agent` | `qwen3.5:9b` | PDF/TXT/Markdown/DOCX creation and document analysis |
| `vision_agent` | `qwen3-vl:8b` | local image/P&ID analysis and structured observations |
| `lightweight_agent` | `llama3.2:1b` | simple local/general requests |

Agent discovery is registry-based and capability-aware. Semantic matching can
rank equivalent descriptions, while artifact, modality, availability, and
policy constraints remain authoritative.

## Tools and artifacts

Implemented local capabilities include workspace listing/search/read, guarded
file creation/editing, Python script creation, command execution, test runs,
Git status/diff, document discovery/extraction/search, OCR/PDF processing,
vision preprocessing/inference, artifact creation/verification, calculator,
and structured audit output. The native MCP adapter exposes the same registry
without creating another executor.

Generated artifacts are workspace-bounded, reopened after writing, and checked
for existence, readability, format validity, non-empty content, and relevant
verification metadata.

## Security model

- Workspace and symlink confinement.
- Central command policy and bounded subprocess execution.
- Process-group termination on timeout.
- Output-size and time limits.
- Approval for file mutations and sensitive commands.
- Network disabled by default.
- Git mutation, deletion, arbitrary shell, credential access, and cloud APIs
  are not enabled by default.
- Visual content is treated as untrusted data.
- Computer-use is disabled until its dedicated security controls and tests are
  implemented.
- Private chain-of-thought is never exposed or persisted; only operational
  events and bounded evidence are recorded.

## Terminal experience

Run:

```bash
source .venv/bin/activate
python cli.py
```

The terminal now provides an AEGIS dashboard with model health, session ID,
Master-first/LangGraph state, live phase updates, approval prompts, response
panels, verification status, repair counts, and output locations.

Interactive commands:

```text
/help      command list
/models    local model health
/network   network and tool counters
/status    AEGIS security/runtime status
/clear     redraw terminal dashboard
/quit      exit
```

Run the API with:

```bash
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Endpoints are documented in `docs/aegis-api.md`; the service uses the same
`Orchestrator.run_master` path as the terminal.

## Validation status

The repository regression suite currently passes with 317 tests and one
intentional skip. The local Qwen-VL acceptance path reaches Ollama with a
validated image payload and returns a structured failure when the installed
local model produces no usable response. AEGIS does not convert that condition
into a false success.

RAG, cloud inference, unrestricted computer-use, and external MCP transport
remain intentionally deferred.

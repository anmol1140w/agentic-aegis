# AEGIS — Sovereign Agent Workbench

> **A self-hosted, air-gapped AI workbench for sensitive industrial, PSU, defence-linked manufacturing, engineering, and government organizations.**

AEGIS is an autonomous, fully-local execution platform engineered to process confidential organizational data—including approval notes, office memos, P&IDs, engineering drawings, financial models, and code—without transmitting any byte across external networks.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites & Dependencies](#3-prerequisites--dependencies)
4. [Installation](#4-installation)
5. [Ollama Setup](#5-ollama-setup)
6. [Model Configuration & Registry](#6-model-configuration--registry)
7. [Running Locally](#7-running-locally)
8. [Running with Docker & Containerization](#8-running-with-docker--containerization)
9. [Code Execution Sandbox](#9-code-execution-sandbox)
10. [Example Workflows & Demonstrations](#10-example-workflows--demonstrations)
11. [Security Model & Boundary Enforcement](#11-security-model--boundary-enforcement)
12. [Network Isolation & Real Monitoring](#12-network-isolation--real-monitoring)
13. [Testing & Verification](#13-testing--verification)
14. [Troubleshooting](#14-troubleshooting)
15. [Documentation Index](#15-documentation-index)

### Clean setup on another system

From the repository root (Python 3.10+):

```bash
bash scripts/setup.sh
source .venv/bin/activate
python cli.py
```

The complete dependency manifest is [requirements.txt](requirements.txt).
For the HTTP service, run `uvicorn app.api.main:app --host 127.0.0.1 --port 8000`.
Install Ollama separately and make the local models listed in
`config/models.yaml` available; setup never downloads models or contacts cloud
services. On Linux, PDF image rendering may also require the system Poppler
package (`pdftoppm`).

---

## 1. Project Purpose

Sensitive enterprises (such as defence PSUs, nuclear and aerospace bodies, heavy engineering firms, and municipal infrastructure authorities) cannot transmit proprietary data to public cloud AI APIs (e.g. OpenAI, Anthropic, Gemini) due to strict statutory, privacy, and national security mandates.

The Workbench solves this by providing:
* **Air-gapped operation:** Complete functionality without public internet connectivity.
* **Master-first orchestration:** The Master discovers capabilities and delegates to bounded local specialists; legacy classifier code is compatibility-only.
* **Deterministic safety & execution:** Tools (calculators, file operations, sandboxed execution) rather than hallucinated LLM operations.
* **Traceability & Auditing:** Full JSONL logging and runtime state tracking with human approval gates for critical actions.

---

## 2. Architecture Overview

The system follows a strict layered topology:

```
                            USER (Terminal CLI)
                                     │
                                     ▼
                            ┌────────────────┐
                            │ Agent Runtime  │ (LangGraph StateGraph)
                            └───────┬────────┘
                                    │
                                    ▼
                         ┌───────────────────────┐
                         │ Task Classifier       │ (Deterministic keyword/heuristic)
                         │ & Model Router        │ (Capability scoring & fallback)
                         └──────────┬────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
        General LLM             Coding LLM           Vision LLM
        (qwen3:8b)         (qwen2.5-coder:7b)     (qwen2.5-vl:7b)
               │                    │                    │
               └────────────────────┼────────────────────┘
                                    │
                                    ▼
                            ┌────────────────┐
                            │  Tool Runtime  │
                            └───────┬────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
         Deterministic         File Operations      Code Sandbox
          Calculator          (Path validated)    (Docker Container)
               │                    │                    │
               └────────────────────┼────────────────────┘
                                    │
                                    ▼
                            ┌────────────────┐
                            │ Structured     │ (SQLite Task Store &
                            │ State & Audit  │  Audit Logger)
                            └────────────────┘
```

Security and observability wrap all operations:
- **Network Isolation:** No external telemetry or cloud fallbacks.
- **Audit Logging:** Structured JSONL records of all task events, models, and tools.
- **Network Monitoring:** OS-level TCP/UDP socket audit using `psutil`.

---

## 3. Prerequisites & Dependencies

### System Requirements
* **OS:** Linux (Ubuntu 20.04+, Debian 11+, RHEL 8+) or macOS
* **Python:** Python 3.10, 3.11, or 3.12
* **Hardware:**
  * Minimum: 16 GB RAM, 4-core CPU (for small quantized models, e.g. `llama3.2:1b`)
  * Recommended: 32 GB RAM, Dedicated NVIDIA GPU with 8GB–16GB VRAM (for 7B/8B parameter models)
* **Ollama:** Version 0.3.0 or higher
* **Docker:** (Optional, for Phase 2 code sandbox) Docker Engine 24.0+

### Core Dependencies
* `langgraph` & `langchain-core`: Graph-based state machine orchestration.
* `langchain-ollama`: Official partner package for local Ollama invocation.
* `pydantic` & `pydantic-settings`: Structured schemas, validation, and configuration.
* `pyyaml`: YAML parsing for model registry and settings.
* `psutil`: Live OS network socket and resource inspection.
* `aiosqlite`: SQLite asynchronous persistence engine.
* `rich`: Terminal UI rendering with tables, colors, and live traces.
* `fastapi` & `uvicorn`: Foundation layer for enterprise extensibility.

---

## 4. Installation

```bash
# 1. Clone or navigate to the repository
cd sih-agent-antigravity

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade build tools
pip install --upgrade pip setuptools wheel

# 4. Install the workbench package in editable mode with development dependencies
pip install -e ".[dev]"
```

For offline or air-gapped environments:
```bash
# Pre-download wheels into a directory on an internet-connected machine:
pip download -d ./wheelhouse ".[dev]"

# On the air-gapped machine:
pip install --no-index --find-links=./wheelhouse --no-build-isolation -e ".[dev]"
```

---

## 5. Ollama Setup

Ensure the Ollama daemon is running locally:

```bash
# Start Ollama service (if not already running as a systemd service)
ollama serve
```

Pull the designated sovereign model suite (or your preferred local equivalents):

```bash
# General Reasoning & Document Processing
ollama pull qwen3:8b

# Code Generation & Debugging
ollama pull qwen2.5-coder:7b

# Vision & Multimodal Analysis
ollama pull qwen2.5-vl:7b

# Lightweight Fallback Model
ollama pull llama3.2:1b
```

Verify model availability:
```bash
ollama list
```

---

## 6. Model Configuration & Registry

The system discovers models dynamically via [`config/models.yaml`](config/models.yaml). You can map any local Ollama tag to logical roles without modifying application source code:

```yaml
models:
  qwen-general:
    provider: ollama
    model: "qwen3:8b"
    capabilities:
      - general
      - reasoning
      - summarization
      - document_analysis
      - artifact_generation
    context_length: 32768
    priority: 10

  qwen-coder:
    provider: ollama
    model: "qwen2.5-coder:7b"
    capabilities:
      - coding
      - debugging
      - calculation
    context_length: 32768
    priority: 10

  qwen-vision:
    provider: ollama
    model: "qwen2.5-vl:7b"
    capabilities:
      - vision
      - multimodal
      - image_analysis
      - document_analysis
    context_length: 32768
    priority: 10

  llama-small:
    provider: ollama
    model: "llama3.2:1b"
    capabilities:
      - general
      - lightweight
    context_length: 8192
    priority: 1
    is_fallback: true
```

---

## 7. Running Locally

Launch the sovereign terminal agent:

```bash
source .venv/bin/activate
python cli.py
# or using the registered console script:
sih-agent
```

### Interactive Commands
Inside the CLI prompt (`You ▶`):
* `/models` — Re-scans and displays availability of models in the registry.
* `/network` — Displays live OS network connection audits and call counters.
* `/quit` or `/exit` — Closes the session and saves audit logs.

---

## 8. Running with Docker & Containerization

For containerized standalone deployments:

```bash
# Build the workbench container
docker build -t sih-workbench:latest .

# Run with host networking to reach Ollama on localhost:11434
docker run -it --rm \
  --network host \
  -v $(pwd)/workspace:/app/workspace \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  sih-workbench:latest
```

---

## 9. Code Execution Sandbox

To satisfy security protocols in industrial and defence environments, generated code is isolated from the host filesystem and OS:

1. **Docker Container Isolation:** Ephemeral containers execute code with `network_disabled=True`.
2. **Resource Constraints:** Hard ceilings on execution memory (`mem_limit='256m'`) and CPU execution quota.
3. **Immutable Root:** Container root is mounted `read_only=True`, with writable access restricted to temporary `tmpfs` mounts.
4. **Dropped Capabilities:** All Linux kernel capabilities dropped (`cap_drop=['ALL']`).

---

## 10. Example Workflows & Demonstrations

### A. General Knowledge / Policy Inquiry
```text
You ▶ Explain the purpose of a Double Block and Bleed (DBB) valve arrangement.
[Task Analysis]
  Task type: general | Modality: text | Model: qwen-general (qwen3:8b)
[Output] Streams response using local qwen-general model...
```

### B. Automated Code Generation & Verification
```text
You ▶ Write a Python script to compute CRC32 checksums for files in the workspace.
[Task Analysis]
  Task type: coding | Modality: text | Model: qwen-coder (qwen2.5-coder:7b)
[Output] Generates verified Python script, saves to workspace/crc32.py...
```

### C. Deterministic Calculation (Tool Routing)
```text
You ▶ Calculate sqrt(144) * 25 + 1024 / 4
[Task Analysis]
  Task type: calculation | Tools: calculator | Model: qwen-coder
[Execution] Calculator tool safely evaluates AST deterministically without LLM math errors.
```

### D. File Operations & Inspection
```text
You ▶ List the files in the workspace and read summary.txt
[Task Analysis]
  Tools: list_files, read_file
[Execution] Paths validated against workspace boundary; file contents returned.
```

---

## 11. Security Model & Boundary Enforcement

* **Strict Path Sandboxing:** The [`security/permissions.py`](security/permissions.py) module enforces workspace confinement. Relative path escapes (`../../etc/passwd`), absolute system paths (`/root/`, `/etc/`, `/proc/`), and dangerous write extensions (`.sh`, `.exe`, `.so`) trigger an immediate `PermissionError_`.
* **Append-Only Audit Logs:** Every task event, tool call, model invocation, and duration is logged to [`logs/audit.jsonl`](logs/audit.jsonl) in structured JSON format. Raw confidential document contents are scrubbed from logs by default.
* **Human-in-the-Loop (HITL) Gate:** Operations tagged with `HIGH` or `CRITICAL` risk levels (such as artifact generation, file overwriting, or code execution) require explicit user authorization.

---

## 12. Network Isolation & Real Monitoring

To guarantee zero data exfiltration during demonstrations:
* The [`security/network.py`](security/network.py) module reads real OS network socket tables via `psutil.net_connections(kind="inet")`.
* Sockets are verified against loopback (`127.0.0.1`, `::1`) and private RFC-1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`).
* The system provides mathematical proof:
  ```text
  Network: external=0  local=2  model_calls=4  tool_calls=2
  ```

---

## 13. Testing & Verification

The test suite covers routing logic, models, security controls, and tools without requiring a running Ollama daemon:

```bash
# Run the complete test suite
python -m pytest tests/ -v

# Run individual verification suites
python -m pytest tests/test_router.py -v    # Classifier, Scoring, Router
python -m pytest tests/test_models.py -v    # Registry, Provider Interface
python -m pytest tests/test_tools.py -v     # AST Calculator, File Operations
python -m pytest tests/test_security.py -v  # Path traversal, Audit, Network monitor
```

---

## 14. Troubleshooting

| Issue | Root Cause | Resolution |
| :--- | :--- | :--- |
| `Config not found: config/models.yaml` | Working directory mismatch | Run `cli.py` from the root of the repository. |
| `No models are available` | Ollama daemon not running or tags missing | Run `ollama serve` and ensure models defined in `config/models.yaml` are pulled via `ollama pull <model>`. |
| `Permission denied: Path escapes workspace boundary` | Tool attempted to access outside `./workspace` | Target paths must remain inside the configured workspace root. |
| `ModuleNotFoundError: No module named 'langgraph'` | Virtual environment not activated or package missing | Activate virtualenv (`source .venv/bin/activate`) and run `pip install -e ".[dev]"`. |
| `External network connections > 0` | Browser or background system processes active | Filter by process PID in `security/network.py` or inspect system sockets via `ss -tunap`. |

---

## 15. Documentation Index

For in-depth specifications, refer to the documents in the [`docs/`](docs/) directory:
* [`docs/architecture.md`](docs/architecture.md) — System layers, state machines, and LangGraph flow.
* [`docs/models.md`](docs/models.md) — Model abstraction, registry schemas, and provider integration.
* [`docs/tools.md`](docs/tools.md) — Tool specifications, risk classifications, and schema definitions.
* [`docs/security.md`](docs/security.md) — Air-gap compliance, sandbox policies, and audit schemas.
* [`docs/workflows.md`](docs/workflows.md) — Workflow definitions, multi-step execution, and human approval.

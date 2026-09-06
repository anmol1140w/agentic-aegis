# AEGIS Architecture — Current Status

AEGIS is the local Master-first workbench. The CLI and FastAPI service use the
same `Orchestrator.run_master` path; the registry selects specialists and the
policy layer authorizes tools. Legacy classifier/router code remains only for
compatibility tests and is not the Master execution path.

```text
CLI or FastAPI → NLP → MasterAgent → AgentRegistry → Specialist
→ Policy/Approval → bounded tools and pipelines → AgentResult
→ Master review/verification/replan → persisted result and trace
```

Available specialists: coding (`qwen2.5-coder:7b`), document/general
(`qwen3.5:9b`), vision (`qwen3-vl:8b`), and lightweight (`llama3.2:1b`).
Document creation supports DOCX, PDF, Markdown, and TXT with artifact
verification. RAG, cloud APIs, arbitrary shell, delete, and Git mutation are
not enabled. See `docs/aegis-api.md` for the HTTP boundary.

---

## 1. System Topology

AEGIS — Sovereign Agent Workbench is structured into distinct, modular, and loosely-coupled layers:

```text
                                 USER
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Terminal CLI   │
                         │    (cli.py)     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Agent Runtime  │  LangGraph StateGraph
                         │ (orchestrator)  │  Checkpoints (Memory/SQLite)
                         └────────┬────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
        ┌─────────────────────┐       ┌─────────────────────┐
        │   Task Classifier   │       │     Model Router    │
        │  & Scoring Engine   │       │   & Model Registry  │
        └─────────────────────┘       └──────────┬──────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
          │   General LLM   │           │   Coding LLM    │           │   Vision LLM    │
          │    (Qwen 3)     │           │ (Qwen2.5-Coder) │           │   (Qwen2.5-VL)  │
          └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
                   │                             │                             │
                   └─────────────────────────────┼─────────────────────────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │    Tool Runtime     │
                                      │   & Tool Registry   │
                                      └──────────┬──────────┘
                                                 │
                   ┌─────────────────────────────┼─────────────────────────────┐
                   ▼                             ▼                             ▼
          ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
          │   Calculator    │           │   File System   │           │  Code Sandbox   │
          │ (AST Evaluator) │           │ (Path Guarded)  │           │(Docker Isolate) │
          └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
                   │                             │                             │
                   └─────────────────────────────┼─────────────────────────────┘
                                                 │
                                                 ▼
                                      ┌─────────────────────┐
                                      │  State Persistence  │
                                      │   & Audit Logging   │
                                      └─────────────────────┘
```

---

## 2. Core Architectural Principles

1. **Deterministic Logic Over LLM Operating System:**
   - Meta-decisions (such as identifying that a user is asking for mathematical calculation or providing an image) are computed via deterministic pattern matchers and heuristics rather than consuming GPU cycles.
   - Calculations and file operations execute directly via deterministic Python tools.

2. **Strict Air-Gap & Sovereignty:**
   - The system assumes `Internet = Unavailable`.
   - All models run through local inference engines (Ollama initially, expandable to vLLM or llama.cpp).
   - Zero telemetry, zero external API dependencies.

3. **Single Orchestrator with Modular Graph Nodes:**
   - Rather than dozens of uncontrolled autonomous agents interacting chaotically, a single LangGraph `StateGraph` orchestrates the task lifecycle:
     - `classify`: Ingests user input and attachments to produce a typed `Task`.
     - `route`: Evaluates model registry candidates against the `Task` to choose the optimal provider.
     - `execute`: Binds active tools to the chosen model and invokes the model with full event streaming.

---

## 3. LangGraph Orchestrator Flow

```text
[START]
   │
   ▼
[classify] ──▶ Analyzes input text, detects modality, task type, tool requirements
   │
   ▼
[route]    ──▶ Scores candidate models in ModelRegistry, chooses primary or fallback
   │
   ▼
[execute]  ──▶ Binds tools (calculator, files), invokes local model, records traces
   │
   ▼
 [END]
```

State is managed by `AgentState`:
- `messages`: Conversation history with LangChain `add_messages` reducer.
- `task`: Strongly typed `Task` object.
- `selected_model`: Selected model key and tag.
- `trace`: Monotonically appended execution steps (`TraceEntry`).
- `errors`: Monotonically appended system or tool errors.

## 4. Master-first migration

### Pre-RAG NLP boundary

User and OCR text first passes through the lightweight deterministic
`NLPPreprocessor`. It preserves `original_text`, produces a conservative
`normalized_text`/`enhanced_prompt`, extracts bounded entities and intent, and
records correction confidence and timing. The Master receives both the
original request and this structured preprocessing context; NLP never selects
agents or executes tools. Raw OCR remains evidence while normalized OCR is
used for reasoning.

```text
USER / OCR → NLP PREPROCESSOR → MASTER → REGISTRY → SPECIALIST → POLICY → TOOLS
```

The insertion point after preprocessing is intentionally reserved for a
future retrieval layer. RAG, embeddings, chunk retrieval, and vector storage
are deferred and are not part of execution today.

All CLI requests use the capability-driven `MasterAgent`; `/master <request>`
remains an explicit diagnostic entry point using the same implementation. The
master plans using registry capabilities, delegates isolated `AgentRequest`
objects, reviews structured `AgentResult` values, persists a master trace, and
returns a structured failure when planning or delegation cannot complete.

```text
USER → CLI → MASTER → REGISTRY → SPECIALIST → POLICY → TOOL → RESULT → REVIEW
                                      │
                                      └── failure → structured Master failure
```

The master never receives unrestricted mutation tools. Specialists use the
existing workspace path guards, command allowlist, approval callbacks, and
network policy. File mutation, arbitrary shell, Git mutation, and RAG remain
explicitly outside the current master capability set. The former classifier
and router are retained only as library compatibility surfaces and are not
invoked by the CLI or Master workflow.

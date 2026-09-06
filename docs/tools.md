# AEGIS Tools — Current Status

Tools are registered centrally in `tools.registry.ToolRegistry` and exposed
only to specialists that have the corresponding allowlist entry. Every tool
returns bounded structured data and is subject to deterministic workspace,
command, network, and approval policy. The Master never receives privileged
mutation tools directly.

Implemented P0 tools include calculator, workspace read/search/list, approved
file creation/editing (including `create_python_script`), approved allowlisted
pytest execution, OCR/PDF processing, document artifact generation, vision
preprocessing, and artifact verification. RAG, web/API, delete, Git mutation,
and unrestricted shell tools remain disabled.

---

## 1. Tool Registry Pattern (`tools/registry.py`)

All workbench tools are managed through a centralized `ToolRegistry`:

* **Metadata Attachment:** Every tool registers its name, description, input/output contract, risk classification (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), approval requirement, timeout, and semantic tags.
* **LangChain Integration:** Tools are decorated with LangChain's `@tool` to expose OpenAI/Ollama-compatible JSON schema definitions to LLMs.
* **Task-Driven Binding:** The orchestrator queries `ToolRegistry` to selectively bind tools relevant to the classified task (e.g., math tools for calculation tasks, file tools for document tasks).

---

## 2. Implemented Tools

### A. Deterministic AST Calculator (`tools/calculator.py`)
* **Purpose:** Provides 100% deterministic mathematical evaluation, eliminating LLM hallucination and arithmetic errors on engineering calculations.
* **Security:** Uses Python's `ast` (Abstract Syntax Tree) module to parse expressions. It explicitly forbids variable assignment, imports, and arbitrary evaluation.
* **Supported Operations:**
  * Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`
  * Math functions: `sqrt`, `log`, `log10`, `log2`, `sin`, `cos`, `tan`, `pi`, `e`, `abs`, `round`, `ceil`, `floor`, `factorial`.

### B. Workspace File Operations (`tools/files.py`)
* **`read_file(file_path: str)`**: Reads file content up to 10MB inside the workspace.
* **`write_file(file_path: str, content: str)`**: Writes content and creates intermediate parent directories. Prohibits writing dangerous executable extensions (`.sh`, `.exe`, `.so`).
* **`list_files(directory: str = ".")`**: Formats structured directory listings with file types and sizes.
* **`create_directory(directory: str)`**: Safely creates directories within the workspace.

---

## 3. Tool Risk Matrix

| Tool | Risk Level | Requires HITL Approval? | Default Timeout |
| :--- | :--- | :--- | :--- |
| `calculator` | LOW | No | 5s |
| `list_files` | LOW | No | 10s |
| `read_file` | LOW | No | 10s |
| `create_directory` | LOW | No | 10s |
| `write_file` | MEDIUM | No (unless overwriting sensitive file) | 15s |
| `python_sandbox` (Phase 2) | HIGH | Yes (or configured per policy) | 30s |
| `artifact_generator` (Phase 4)| HIGH | Yes (Approval Note creation) | 60s |

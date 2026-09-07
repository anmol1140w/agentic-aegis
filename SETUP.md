# AEGIS setup guide

This is the complete setup path for Linux, macOS, and Windows. AEGIS is
local-first: Python runs the workbench and Ollama runs models on the same
machine. No cloud key is required for the normal path.

## 1. Host prerequisites

Install Git, Python 3.10 or newer, and Ollama. Python 3.11 or 3.12 is
recommended.

Linux Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

Optional Linux tools: `cmake`, `g++`, `poppler-utils`, and `bubblewrap`.

macOS:

```bash
brew install git python cmake
```

Windows PowerShell: install Python from [python.org](https://www.python.org/downloads/windows/),
Git, Ollama, and optionally CMake plus Visual Studio Build Tools with the C++
workload. Enable “Add Python to PATH” and open a new PowerShell window.

Verify on every platform:

```text
python --version
git --version
ollama --version
```

Docker Desktop is optional. Bubblewrap is Linux-only. The portable process
sandbox is the fallback on all platforms.

## 2. Download and install

Linux/macOS:

```bash
git clone <repository-url> aegis
cd aegis
bash scripts/setup.sh
source .venv/bin/activate
```

Windows PowerShell:

```powershell
git clone <repository-url> aegis
cd aegis
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
git clone <repository-url> aegis
cd aegis
scripts\setup.cmd
.venv\Scripts\activate.bat
```

The scripts install from `pyproject.toml`; it is the single dependency source.
There are intentionally no `requirements*.txt` files. Optional groups are
`dev`, `doc`, `artifacts`, `ocr`, and `sandbox`:

```bash
AEGIS_INSTALL_EXTRAS='dev,doc,artifacts,sandbox' bash scripts/setup.sh
# Linux OCR, only when needed:
AEGIS_INSTALL_EXTRAS='dev,ocr' bash scripts/setup.sh
```

PowerShell:

```powershell
.\scripts\setup.ps1 -Extras 'dev,doc,artifacts,sandbox'
```

## 3. Install local models

Start Ollama and pull the configured local aliases:

```bash
ollama serve
ollama pull qwen3.5:9b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-vl:8b
ollama list
```

Model tags live in `config/models.yaml`, not in routing logic. Change that file
when using different local tags.

## 4. Run and test

```bash
python cli.py
python -m pytest -q
```

Use `.venv/bin/python` on Linux/macOS or `.venv\Scripts\python.exe` on Windows
if the environment is not activated. Try `create a Python file about linked
lists and run its tests` in the CLI.

## 5. Optional API and sandboxes

```bash
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

The default process-group sandbox is portable. Docker mode requires Docker:

```bash
export AEGIS_SANDBOX_BACKEND=docker
export AEGIS_CONTAINER_IMAGE=python:3.12-slim
```

Linux Bubblewrap mode:

```bash
export AEGIS_SANDBOX_BACKEND=bwrap
```

Unavailable optional backends fail with a diagnostic; AEGIS never reports a
command as sandboxed when the backend did not run.

## 6. Directory contract

Active source is in `runtime/`, `tools/`, `models/`, `security/`, `storage/`,
`pipeline/`, `app/`, `tests/`, `native/`, `config/`, and `docs/`. The controlled
user area is `workspace/`:

- `fixtures/` — reusable offline inputs and tests.
- `agent_test/` — agent-facing contract tests.
- `artifacts/` — verified reusable outputs and provenance.
- `executions/` — bounded command evidence.
- `outputs/` — generated per-run output; do not commit it.

Virtual environments, caches, build products, audit logs, model files, and
wheel caches are intentionally excluded from the repository.

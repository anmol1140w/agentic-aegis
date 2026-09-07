# AEGIS local setup

This guide installs AEGIS — Sovereign Agent Workbench on clean Linux, Windows,
or macOS hosts. AEGIS runs its application and inference locally; Ollama models
must be installed separately by the operator. No editable local path, virtual
environment, wheelhouse, build directory, or generated workspace output
should be committed or shared.

The standard process sandbox works on all three platforms. Docker is optional
and works wherever Docker Engine/Desktop is available. Bubblewrap is optional
and Linux-only.

## 1. Prerequisites by platform

Install Python 3.10 or newer (Python 3.12 is supported), Git, and Ollama.
Python 3.11 or 3.12 is recommended for the broadest dependency wheel support.

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git poppler-utils
```

For Fedora, use `sudo dnf install python3 python3-pip git poppler-utils`.

### Windows (PowerShell)

Install Python from [python.org](https://www.python.org/downloads/windows/),
Git for Windows, and Ollama for Windows. In PowerShell, verify:

```powershell
py --version
git --version
ollama --version
```

Install Poppler only if PDF rendering through `pdf2image` is required. Download
a trusted Windows Poppler build, extract it, and add its `Library\bin` directory
to `PATH`. PyMuPDF is already included and is the preferred PDF path.

### macOS

Install Homebrew, then:

```bash
brew install python git poppler ollama
python3 --version
ollama --version
```

Network access is needed only to install packages/models; normal AEGIS
inference and document processing use local services.

## 2. Create the Python environment

### Linux and macOS

```bash
git clone <repository-url> aegis
cd aegis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-dev.txt
```

The supported repeatable installer is:

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

It creates `.venv` and installs the editable package with development tools.
To include the Docker Python SDK too, use
`AEGIS_INSTALL_EXTRAS='dev,sandbox' bash scripts/setup.sh`.

### Windows PowerShell

```powershell
git clone <repository-url> aegis
Set-Location aegis
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

For Docker Python SDK support:

```powershell
.\scripts\setup.ps1 -Extras 'dev,sandbox'
```

Command Prompt users can run `scripts\setup.cmd` instead. It installs the
standard development environment; use PowerShell when the Docker extra is
needed.

If PowerShell blocks activation for the current user, use the documented
Python execution path without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

## 3. Optional OCR and PDF host dependencies

The base package installs PaddleOCR and PaddlePaddle. If your platform or
Python version cannot resolve the compatible PaddlePaddle CPU wheel, install
the platform-specific wheel from the official CPU index, then install
PaddleOCR:

```bash
python -m pip install paddleocr==3.7.0
python -m pip install paddlepaddle==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

Verify the installation:

```bash
python -c "import paddle; print(paddle.__version__)"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
```

The first OCR run may download PaddleX models into the user's local cache.
After they are cached, OCR can operate without internet access.

On Windows PowerShell, use one line per command:

```powershell
python -m pip install paddleocr==3.7.0
python -m pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

## 4. Configure and verify Ollama

Ensure Ollama is running locally, then install only the models listed in
`config/models.yaml` (model downloads are large and are not performed by the
setup script):

```bash
ollama serve
ollama list
ollama pull qwen3.5:9b
ollama pull qwen2.5-coder:7b
ollama pull qwen3-vl:8b
ollama pull llama3.2:1b
```

On Windows, start **Ollama** from the Start menu (or run `ollama serve` in a
terminal), then run the same `ollama list`/`ollama pull` commands in
PowerShell. On macOS, `brew services start ollama` can keep the local service
running.

The application uses the local endpoint configured by the model registry; it
does not require cloud API keys.

## 5. Optional execution sandboxes

The default `process` backend needs no extra runtime. Check it with `/sandbox`
after starting the CLI.

For Docker:

1. Install Docker Engine on Linux or Docker Desktop on macOS/Windows.
2. Start the Docker daemon/Desktop application.
3. Pull the image used by AEGIS:

```bash
docker pull python:3.12-slim
AEGIS_SANDBOX_BACKEND=docker python cli.py
```

PowerShell:

```powershell
docker pull python:3.12-slim
$env:AEGIS_SANDBOX_BACKEND = "docker"
python cli.py
```

The image is not stored in Git and is not downloaded automatically. AEGIS
returns `sandbox_unavailable` if Docker or the image is missing. The container
has no network, drops capabilities, and mounts only the workspace read-write.

For Bubblewrap, install `bwrap` from the Linux distribution packages and run:

```bash
sudo apt-get install -y bubblewrap   # Debian/Ubuntu
AEGIS_SANDBOX_BACKEND=bwrap python cli.py
```

There is no Bubblewrap backend on native Windows or macOS; use the process or
Docker backend there.

## 6. Run the terminal client

```bash
source .venv/bin/activate
python cli.py
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python cli.py
```

### Terminal CLI workflow

After startup, the prompt is shown as `You ▶`. Enter a natural-language task;
the Master Agent handles capability discovery, specialist delegation, policy,
verification, and the final response:

```text
You ▶ list files in the workspace
You ▶ Inspect verify_runs.py and identify one potential bug. Do not modify files.
You ▶ create txt about agentic AI
You ▶ Analyze workspace/fixtures/codex_multi_page_inspection.pdf
You ▶ Analyze test_image.jpg
```

For coding mutations, the terminal pauses for explicit approval before a file
is created/edited and again before an approved test command runs:

```text
⚠  Approve: create_python_script race_stats.py?
   content: 640 chars
   [y/N] > y
```

Use `y`/`yes` to approve or press Enter/answer `n` to deny. A denied action is
reported as a structured failure and does not modify the workspace. The CLI
also supports these commands:

```text
/models    # local model availability
/network   # local/external connection report
/quit      # exit the session
```

Generated documents and run evidence are saved under
`workspace/outputs/run_<id>/`, including `result.txt`, `metadata.json`, and
`trace.json`. The CLI prints the output directory when a request completes.

To run a deterministic inspection pipeline directly (without an interactive
prompt), use:

```bash
python cli.py inspect-report \
  --input workspace/fixtures/codex_multi_page_inspection.pdf \
  --output-dir workspace/outputs/manual_inspection
```

On Windows PowerShell:

```powershell
python cli.py inspect-report `
  --input workspace/fixtures/codex_multi_page_inspection.pdf `
  --output-dir workspace/outputs/manual_inspection
```

Useful examples:

```text
list files in the workspace
Inspect verify_runs.py and identify one potential bug. Do not modify files.
create txt about agentic AI
Analyze workspace/fixtures/codex_multi_page_inspection.pdf
```

Interactive commands include `/models`, `/network`, and `/quit`.

## 7. Run the FastAPI service

In a second terminal:

```bash
cd aegis
source .venv/bin/activate
uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
Set-Location aegis
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Check health and submit a task:

```bash
curl http://127.0.0.1:8000/api/health
curl -X POST http://127.0.0.1:8000/api/tasks \
  -H 'content-type: application/json' \
  -d '{"request":"list files in the workspace","files":[],"options":{}}'
```

Use the returned execution ID:

```bash
curl http://127.0.0.1:8000/api/tasks/<execution_id>
curl -N http://127.0.0.1:8000/api/tasks/<execution_id>/events
```

The service is intended for a Django backend over HTTP; Django should not
import AEGIS runtime classes directly.

## 8. OCR and tests

Run the OCR smoke checks with the repository's available fixtures, then the
full test suite:

```bash
python -m pytest -q tests/test_ocr.py
python -m pytest -q
```

Live Ollama tests are marked separately and require the corresponding local
model. Do not run them in an air-gapped environment until models are cached.

## 9. Sharing and cleanup

Share source, `config/`, `docs/`, `requirements*.txt`, `pyproject.toml`, and
`SETUP.md`. Do not share `.venv/`, `wheelhouse/`, `build/`, `dist/`,
`__pycache__/`, logs, secrets, or generated `workspace/outputs/`.

For an optional environment snapshot, use `python -m pip freeze` into a local
lock file and remove any `-e /absolute/local/path` entry before sharing it.
RAG/embeddings remain deferred; no vector store is required for this setup.

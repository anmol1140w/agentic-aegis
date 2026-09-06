#!/usr/bin/env bash
set -euo pipefail

# Clean, repeatable local setup for AEGIS. No cloud service or model download
# is performed here; Ollama models are installed separately by the operator.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

echo "AEGIS environment ready. Activate with: source $VENV_DIR/bin/activate"
echo "Ensure Ollama is installed separately and configured with config/models.yaml."

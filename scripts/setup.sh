#!/usr/bin/env bash
set -euo pipefail

# Clean, repeatable local setup for AEGIS on Linux and macOS. No cloud service
# or model download is performed here; Ollama models are installed separately.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_EXTRAS="${AEGIS_INSTALL_EXTRAS:-dev}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e ".[${INSTALL_EXTRAS}]"

echo "AEGIS environment ready. Activate with: source $VENV_DIR/bin/activate"
echo "Optional extras: AEGIS_INSTALL_EXTRAS='dev,sandbox' bash scripts/setup.sh"
echo "Install Ollama separately and pull the tags in config/models.yaml."

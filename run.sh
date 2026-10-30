#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 or later is required. Install it, or run with PYTHON_BIN=/path/to/python3.11."
  exit 1
fi

if [ ! -d "$ENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$ENV_DIR/bin/streamlit" run "$PROJECT_DIR/app.py"

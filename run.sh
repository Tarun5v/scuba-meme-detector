#!/usr/bin/env bash
# Launch the scuba meme detector.
# Run from the project root, or from anywhere (it resolves its own path).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
    echo "[!] Virtual environment not found. Run:"
    echo "    /opt/homebrew/bin/python3.11 -m venv .venv"
    echo "    .venv/bin/pip install -r requirements.txt"
    exit 1
fi

exec "$PY" src/main.py "$@"
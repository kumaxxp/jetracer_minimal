#!/bin/bash
set -euo pipefail

PROJECT_DIR="/home/jetson/jetracer_minimal"
VENV_DIR="$PROJECT_DIR/.venv"

echo "Starting JetRacer Minimal..."
cd "$PROJECT_DIR"

if pgrep -f "python3 main.py" >/dev/null; then
    echo "Application is already running!"
    exit 1
fi

if [ -d "$VENV_DIR" ]; then
    # shellcheck source=/home/jetson/jetracer_minimal/.venv/bin/activate
    . "$VENV_DIR/bin/activate"
else
    echo "Warning: virtual environment not found at $VENV_DIR" >&2
fi

exec python3 main.py

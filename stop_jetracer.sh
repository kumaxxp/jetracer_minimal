#!/bin/bash
set -euo pipefail

echo "Stopping JetRacer Minimal..."
if pkill -f "python3 main.py"; then
    echo "Application stopped."
else
    echo "Application was not running."
fi

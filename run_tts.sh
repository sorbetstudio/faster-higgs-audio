#!/bin/bash
# Wrapper script to run TTS with proper venv

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use the venv python
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python3"

# Check if venv exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "Error: Virtual environment not found at $PYTHON_PATH"
    echo "Please run: uv pip install -r requirements.txt && uv pip install -e ."
    exit 1
fi

# Run the TTS script with all arguments
"$PYTHON_PATH" "$SCRIPT_DIR/tts_script.py" "$@"
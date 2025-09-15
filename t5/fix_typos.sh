#!/bin/bash
# Simple wrapper script for typo correction

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if input is provided
if [ -z "$1" ]; then
    echo "Usage: $0 \"text with typos\""
    echo "   or: echo \"text\" | $0"
    echo ""
    echo "Examples:"
    echo "  $0 \"I beleive this is teh answr\""
    echo "  echo \"The qucik brown fox\" | $0"
    exit 1
fi

# Run the Python script
cd "$SCRIPT_DIR"
python3 typo_corrector.py "$1"
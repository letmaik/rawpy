#!/bin/bash
# Install a specific Python version (Ubuntu only, via deadsnakes PPA)
# and create a venv with it.
#
# Usage:
#   bash scripts/setup_python.sh 3.12
#   bash scripts/setup_python.sh 3.9
#
# After this, run the normal setup:
#   bash scripts/setup_agent_env.sh
#
# The venv at .venv will use the requested Python version.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

PYTHON_VERSION="${1:?Usage: $0 <python-version>  (e.g. 3.12)}"

# Validate format
if ! [[ "$PYTHON_VERSION" =~ ^3\.[0-9]+$ ]]; then
    echo "ERROR: Version must be in X.Y format (e.g. 3.12), got: $PYTHON_VERSION"
    exit 1
fi

PYTHON_BIN="python${PYTHON_VERSION}"

# Check if already available
if command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Python $PYTHON_VERSION already installed: $($PYTHON_BIN --version)"
else
    echo "Python $PYTHON_VERSION not found, installing via deadsnakes PPA..."
    echo "(This requires sudo on Ubuntu/Debian)"
    echo ""

    if ! command -v apt-get &> /dev/null; then
        echo "ERROR: apt-get not found. This script only supports Ubuntu/Debian."
        echo "Install Python $PYTHON_VERSION manually, then re-run."
        exit 1
    fi

    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${PYTHON_BIN}" "${PYTHON_BIN}-venv" "${PYTHON_BIN}-dev"

    echo "Installed: $($PYTHON_BIN --version)"
fi

# Remove existing venv if it uses a different Python
if [ -d "$VENV_DIR" ]; then
    CURRENT=$("$VENV_DIR/bin/python" --version 2>/dev/null | awk '{print $2}' | cut -d. -f1,2)
    if [ "$CURRENT" = "$PYTHON_VERSION" ]; then
        echo "Existing .venv already uses Python $PYTHON_VERSION, keeping it."
        exit 0
    fi
    echo "Removing existing .venv (Python $CURRENT) to replace with $PYTHON_VERSION..."
    rm -rf "$VENV_DIR"
fi

echo "Creating .venv with Python $PYTHON_VERSION..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

echo ""
echo "Done. .venv now uses $("$VENV_DIR/bin/python" --version)."
echo ""
echo "Next step — run the full environment setup:"
echo "  bash scripts/setup_agent_env.sh"

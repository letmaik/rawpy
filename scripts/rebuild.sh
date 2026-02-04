#!/bin/bash
# Rebuilds the rawpy Cython extension.
# Use this after changing .pyx files, C++ source, or headers.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

# Require venv
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: .venv not found."
    echo "Run 'bash scripts/setup_agent_env.sh' first."
    exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

echo "Rebuilding rawpy..."
# --no-build-isolation: reuses current env's numpy/cython (faster)
# -e: editable install (.py changes apply immediately)
pip install --no-build-isolation -e . -q

# Verify the build succeeded
echo ""
if python -c "import rawpy; print(f'rawpy {rawpy.__version__} rebuilt successfully')"; then
    exit 0
else
    echo "ERROR: Build completed but import failed."
    exit 1
fi

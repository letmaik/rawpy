#!/bin/bash
# Switch the active venv to a specific numpy version.
#
# Usage:
#   bash scripts/setup_numpy.sh 2.0.2
#   bash scripts/setup_numpy.sh '2.1.*'
#
# After this, use pytest / mypy / any command as normal.
# The Cython extension does NOT need rebuilding — NumPy ABI 2.0+ is
# forward-compatible at runtime, and the stubs ship with numpy itself.

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

NUMPY_VERSION="${1:?Usage: $0 <numpy-version>  (e.g. 2.0.2 or '2.0.*')}"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: .venv not found. Run 'bash scripts/setup_agent_env.sh' first."
    exit 1
fi
source "$VENV_DIR/bin/activate"

BEFORE=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "none")
echo "Current numpy: $BEFORE"
echo "Installing numpy==$NUMPY_VERSION ..."
pip install "numpy==$NUMPY_VERSION" -q
AFTER=$(python -c "import numpy; print(numpy.__version__)")
echo "Now using numpy: $AFTER"

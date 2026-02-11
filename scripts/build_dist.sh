#!/bin/bash
# Build sdist and wheel artifacts for local testing.
#
# This uses build isolation (like CI), so the result closely matches
# what gets published to PyPI. Output goes to dist/.
#
# Usage:
#   bash scripts/build_dist.sh
#   RAWPY_USE_SYSTEM_LIBRAW=1 bash scripts/build_dist.sh
#     Builds the wheel against the system libraw instead of the bundled
#     source. The sdist is unaffected.
#
# To build with a different Python version:
#   bash scripts/setup_python.sh 3.12
#   bash scripts/build_dist.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

# Require venv
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: .venv not found."
    echo "Run 'bash scripts/setup_dev_env.sh' first."
    exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

# Ensure 'build' package is available
if ! python -m build --help &>/dev/null; then
    echo "Installing 'build' package..."
    pip install build -q
fi

# Clean previous artifacts
rm -rf dist/

# Ensure submodules are initialized (needed for sdist to include LibRaw source)
if [ ! -f "external/LibRaw/README.md" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

echo "=== Building sdist + wheel ==="
echo ""
python -m build

echo ""
echo "=== Build complete ==="
echo "Artifacts in dist/:"
ls -lh dist/

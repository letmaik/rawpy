#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

# Auto-activate venv if not active
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "Activating virtual environment..."
        source "$VENV_DIR/bin/activate"
    else
        echo "ERROR: No virtual environment found at $VENV_DIR"
        echo "Run 'bash scripts/setup_agent_env.sh' first."
        exit 1
    fi
fi

echo "=== Agent Environment Check ==="
echo ""

# Step 1: Verify rawpy can be imported
echo "1. Checking rawpy import..."
if ! python -c "import rawpy; print(f'   rawpy {rawpy.__version__}')"; then
    echo "   FAILED: Cannot import rawpy"
    echo ""
    echo "   The Cython extension may not be built."
    echo "   Run: bash scripts/rebuild.sh"
    exit 1
fi
echo "   OK"
echo ""

# Step 2: Type checking
echo "2. Checking types (mypy)..."
cd "$PROJECT_ROOT"
if python -m mypy rawpy --no-error-summary 2>/dev/null; then
    echo "   OK"
else
    echo "   WARNINGS (non-fatal, review output above)"
fi
echo ""

# Step 3: Quick runtime test
echo "3. Running quick test..."
if pytest test/test_basic.py::testFileOpenAndPostProcess -v --tb=short 2>/dev/null; then
    echo "   OK"
else
    echo "   FAILED"
    exit 1
fi

echo ""
echo "=== All Checks Passed ==="

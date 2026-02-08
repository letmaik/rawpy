#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

# Optional: specify a Python version (e.g. 3.12)
PYTHON_VERSION="${1:-}"

echo "=== Agent Environment Setup ==="
echo ""

# If a specific Python version was requested, install it first
if [ -n "$PYTHON_VERSION" ]; then
    echo "Requested Python $PYTHON_VERSION"
    bash "$SCRIPT_DIR/setup_python.sh" "$PYTHON_VERSION"
    echo ""
fi

# Check system dependencies first
echo "Checking system dependencies..."

# Check for cmake
if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake is not installed."
    echo "  Ubuntu/Debian: sudo apt install cmake"
    echo "  macOS:         brew install cmake"
    echo "  Fedora:        sudo dnf install cmake"
    exit 1
fi
echo "  cmake: $(cmake --version | head -1)"

# Check for C++ compiler
if command -v g++ &> /dev/null; then
    echo "  C++ compiler: g++ $(g++ --version | head -1)"
elif command -v clang++ &> /dev/null; then
    echo "  C++ compiler: clang++ $(clang++ --version | head -1)"
else
    echo "ERROR: No C++ compiler found (g++ or clang++)."
    echo "  Ubuntu/Debian: sudo apt install g++"
    echo "  macOS:         xcode-select --install"
    exit 1
fi

echo ""

# Create or reuse venv
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Using existing virtual environment."
fi

source "$VENV_DIR/bin/activate"

echo "  Python: $(python --version)"
echo "  pip: $(pip --version | cut -d' ' -f1-2)"
echo ""

echo "Upgrading pip..."
pip install --upgrade pip -q

echo "Initializing git submodules..."
cd "$PROJECT_ROOT"
git submodule update --init --recursive

echo "Installing Python dependencies..."
# This includes setuptools since --no-build-isolation skips build-system.requires.
pip install -r "$PROJECT_ROOT/dev-requirements.txt" -q

echo "Building and installing rawpy (this may take a minute)..."
# Delete stale _rawpy.cpp so cythonize() regenerates it (see AGENTS.md)
rm -f rawpy/_rawpy.cpp
pip install -e "$PROJECT_ROOT" --no-build-isolation -q

# Verify the installation
echo ""
echo "Verifying installation..."
if python -c "import rawpy; print(f'  rawpy {rawpy.__version__} installed successfully')" 2>/dev/null; then
    echo ""
    echo "=== Setup Complete ==="
    echo ""
    echo "To activate this environment:"
    echo "  source .venv/bin/activate"
    echo ""
    echo "Quick verification:"
    echo "  bash scripts/agent_check.sh"
else
    echo ""
    echo "ERROR: rawpy import failed after installation."
    echo "Check the build output above for errors."
    exit 1
fi

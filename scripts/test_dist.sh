#!/bin/bash
# Install a built artifact (wheel or sdist) into a clean temporary venv
# and run the test suite against it. This validates that the package works
# as an end-user would experience it — no editable install, no source tree
# on sys.path.
#
# Usage:
#   bash scripts/test_dist.sh sdist              # test the sdist
#   bash scripts/test_dist.sh wheel              # test the wheel
#   bash scripts/test_dist.sh sdist 2.0.2        # test sdist with numpy 2.0.2
#   bash scripts/test_dist.sh wheel 2.0.2        # test wheel with numpy 2.0.2
#
#   RAWPY_USE_SYSTEM_LIBRAW=1 bash scripts/test_dist.sh sdist
#     Install the sdist using system libraw and verify linkage.
#     Prerequisites: sudo apt install libraw-dev pkg-config
#     Note: Requires system LibRaw >= 0.21. Ubuntu 22.04 ships 0.20.2
#     which is too old. Use Ubuntu 24.04+ or install from source.
#
# The script creates a temporary venv (.venv-test) using the same Python
# as .venv, installs the artifact, runs pytest, and cleans up on exit.
#
# To test with a different Python version:
#   bash scripts/setup_python.sh 3.12   # recreates .venv with Python 3.12
#   bash scripts/build_dist.sh           # build artifacts
#   bash scripts/test_dist.sh sdist      # test it
#
# Prerequisites:
#   bash scripts/build_dist.sh   # build the artifact(s) first

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_VENV="$PROJECT_ROOT/.venv-test"
DIST_DIR="$PROJECT_ROOT/dist"

ARTIFACT_TYPE="${1:?Usage: $0 <sdist|wheel> [numpy-version]}"
NUMPY_VERSION="${2:-}"      # optional, e.g. "2.0.2"
VENV_DIR="$PROJECT_ROOT/.venv"

# Use the same Python as .venv (set up by setup_python.sh / setup_dev_env.sh)
if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
else
    echo "ERROR: .venv not found. Run 'bash scripts/setup_dev_env.sh' first."
    exit 1
fi

# --- Validation ---

case "$ARTIFACT_TYPE" in
    sdist|wheel) ;;
    *)
        echo "Usage: $0 <sdist|wheel> [numpy-version]"
        echo ""
        echo "  sdist  — install and test the .tar.gz source distribution"
        echo "  wheel  — install and test the .whl wheel"
        echo ""
        echo "  numpy-version — optional, e.g. 2.0.2 or '2.1.*'"
        exit 1
        ;;
esac

if [ ! -d "$DIST_DIR" ]; then
    echo "ERROR: dist/ directory not found."
    echo "Run 'bash scripts/build_dist.sh' first."
    exit 1
fi

# --- Find the artifact ---

find_artifact() {
    local pattern="$1"
    local matches
    matches=$(find "$DIST_DIR" -maxdepth 1 -name "$pattern" | head -1)
    echo "$matches"
}

case "$ARTIFACT_TYPE" in
    sdist)
        ARTIFACT=$(find_artifact "rawpy-*.tar.gz")
        if [ -z "$ARTIFACT" ]; then
            echo "ERROR: No sdist found in dist/. Run 'bash scripts/build_dist.sh'."
            exit 1
        fi
        ;;
    wheel)
        ARTIFACT=$(find_artifact "rawpy-*.whl")
        if [ -z "$ARTIFACT" ]; then
            echo "ERROR: No wheel found in dist/. Run 'bash scripts/build_dist.sh'."
            exit 1
        fi
        ;;
esac

echo "=== Testing Distribution Artifact ==="
echo "  Artifact: $(basename "$ARTIFACT")"
echo "  Python:   $($PYTHON_BIN --version 2>&1) (from .venv)"
if [ -n "$NUMPY_VERSION" ]; then
    echo "  NumPy:    $NUMPY_VERSION"
fi
if [ "$RAWPY_USE_SYSTEM_LIBRAW" = "1" ]; then
    echo "  LibRaw:   system (RAWPY_USE_SYSTEM_LIBRAW=1)"
fi
echo ""

# --- Cleanup handler ---

cleanup() {
    if [ -d "$TEST_VENV" ]; then
        echo ""
        echo "Cleaning up test venv..."
        rm -rf "$TEST_VENV"
    fi
}
trap cleanup EXIT

# --- Create clean test venv ---

echo "Creating clean test venv..."
rm -rf "$TEST_VENV"
"$PYTHON_BIN" -m venv "$TEST_VENV"
source "$TEST_VENV/bin/activate"

pip install --upgrade pip -q

# --- Install the artifact ---

echo "Installing $(basename "$ARTIFACT")..."
if [[ "$ARTIFACT" == *.tar.gz ]]; then
    # sdist: pip will build from source (with build isolation)
    # This tests that the sdist contains everything needed to build.
    pip install "$ARTIFACT[test]" --verbose
else
    # wheel: direct install
    pip install "$ARTIFACT[test]" -q
fi

# --- Pin numpy version if requested ---

if [ -n "$NUMPY_VERSION" ]; then
    echo "Installing numpy==$NUMPY_VERSION..."
    pip install "numpy==$NUMPY_VERSION" -q
fi

# --- Verify import ---

echo ""
echo "Verifying rawpy import..."
# Run from a temp directory so Python doesn't pick up the source tree's rawpy/
VERIFY_DIR=$(mktemp -d)
cd "$VERIFY_DIR"

RAWPY_VERSION=$(python -c "import rawpy; print(rawpy.__version__)")
echo "  rawpy $RAWPY_VERSION imported successfully"

RAWPY_LOCATION=$(python -c "import rawpy; print(rawpy.__file__)")
echo "  Location: $RAWPY_LOCATION"

# Sanity check: rawpy should NOT be loaded from the source tree
if [[ "$RAWPY_LOCATION" == "$PROJECT_ROOT/rawpy/"* ]]; then
    echo "  ERROR: rawpy is loaded from the source tree, not from the installed package."
    echo "  This means the test would not validate the artifact."
    cd "$PROJECT_ROOT"
    rm -rf "$VERIFY_DIR"
    exit 1
fi

NUMPY_ACTUAL=$(python -c "import numpy; print(numpy.__version__)")
echo "  numpy $NUMPY_ACTUAL"

# --- Verify system libraw linkage (Linux only) ---

if [ "$RAWPY_USE_SYSTEM_LIBRAW" = "1" ]; then
    echo ""
    echo "Verifying system libraw linkage..."
    python -c "
import rawpy._rawpy as _rawpy
import os, subprocess, sys

ext_path = _rawpy.__file__
pkg_dir = os.path.dirname(ext_path)

# Check 1: No bundled libraw_r.so in the package directory
bundled = [f for f in os.listdir(pkg_dir) if f.startswith('libraw_r.so')]
if bundled:
    print(f'  FAIL: Found bundled libraw files: {bundled}')
    sys.exit(1)
print('  OK: No bundled libraw_r.so in package directory')

# Check 2: ldd shows system libraw, not a local path
result = subprocess.run(['ldd', ext_path], capture_output=True, text=True)
libraw_lines = [l.strip() for l in result.stdout.splitlines() if 'libraw_r' in l]
if not libraw_lines:
    print('  FAIL: libraw_r.so not found in ldd output')
    sys.exit(1)
for line in libraw_lines:
    print(f'  ldd: {line}')
    if pkg_dir in line:
        print('  FAIL: libraw_r.so resolves to the package directory')
        sys.exit(1)
print('  OK: libraw_r.so links to system library')
"
fi

cd "$PROJECT_ROOT"
rm -rf "$VERIFY_DIR"

# --- Run tests ---

echo ""
echo "Running tests..."
# Run from a temp directory so Python doesn't pick up the source tree's rawpy/
WORK_DIR=$(mktemp -d)
cd "$WORK_DIR"

pytest --verbosity=3 -s "$PROJECT_ROOT/test"

TEST_EXIT=$?

cd "$PROJECT_ROOT"
rm -rf "$WORK_DIR"

echo ""
if [ $TEST_EXIT -eq 0 ]; then
    echo "=== All tests passed ==="
else
    echo "=== Tests FAILED (exit code: $TEST_EXIT) ==="
    exit $TEST_EXIT
fi

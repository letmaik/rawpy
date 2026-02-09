#!/bin/bash
# Test sdist: install from source in a clean venv and run the test suite.
# This validates that the sdist contains everything needed to build.
#
# When RAWPY_USE_SYSTEM_LIBRAW=1 is set, the sdist is built against the
# system libraw and linkage is verified (no bundled libraw_r.so, ldd
# points to system library).
set -e -x

PYTHON_BIN="python${PYTHON_VERSION}"

# Install system build dependencies (C++ compiler is already available)
sudo apt-get update -q
sudo apt-get install -y -q g++

# Create a clean venv
${PYTHON_BIN} -m venv sdist-test-env
source sdist-test-env/bin/activate
python -m pip install --upgrade pip

# Install the sdist (pip will build from source with build isolation)
# RAWPY_USE_SYSTEM_LIBRAW is inherited from the environment if set.
SDIST=$(ls dist/rawpy-*.tar.gz | head -1)
pip install "${SDIST}[test]"

# Run tests from a temp directory to avoid importing from the source tree
mkdir tmp_for_test
pushd tmp_for_test

# Verify system libraw linkage when applicable
if [ "$RAWPY_USE_SYSTEM_LIBRAW" = "1" ]; then
    python -c "
import rawpy._rawpy as _rawpy
import os, subprocess, sys

ext_path = _rawpy.__file__
pkg_dir = os.path.dirname(ext_path)
print(f'Extension: {ext_path}')

# No bundled libraw_r.so in the package directory
bundled = [f for f in os.listdir(pkg_dir) if f.startswith('libraw_r.so')]
if bundled:
    print(f'FAIL: Found bundled libraw files: {bundled}')
    sys.exit(1)
print('OK: No bundled libraw_r.so in package directory')

# ldd shows system libraw, not a local path
result = subprocess.run(['ldd', ext_path], capture_output=True, text=True)
libraw_lines = [l.strip() for l in result.stdout.splitlines() if 'libraw_r' in l]
if not libraw_lines:
    print('FAIL: libraw_r.so not found in ldd output')
    sys.exit(1)
for line in libraw_lines:
    print(f'ldd: {line}')
    if pkg_dir in line:
        print('FAIL: libraw_r.so resolves to the package directory')
        sys.exit(1)
print('OK: libraw_r.so links to system library')
"
fi

pytest --verbosity=3 -s ../test
popd

deactivate

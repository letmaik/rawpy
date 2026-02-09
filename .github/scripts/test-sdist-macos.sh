#!/bin/bash
# Test sdist: install from source in a clean venv and run the test suite.
# This validates that the sdist contains everything needed to build on macOS.
set -e -x

# Create a clean venv
python${PYTHON_VERSION} -m venv sdist-test-env
source sdist-test-env/bin/activate
python -m pip install --upgrade pip

# Install the sdist (pip will build from source with build isolation)
SDIST=$(ls dist/rawpy-*.tar.gz | head -1)
pip install "${SDIST}[test]"

# Run tests from a temp directory to avoid importing from the source tree
mkdir tmp_for_test
pushd tmp_for_test
pytest --verbosity=3 -s ../test
popd

deactivate

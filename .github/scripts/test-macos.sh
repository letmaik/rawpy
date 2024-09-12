#!/bin/bash
set -e -x
PYVER=${PYTHON_VERSION//.}

# Create venv and activate
VENV=testsuite
python -m venv env/$VENV
source env/$VENV/bin/activate

# Install rawpy
pip install dist/rawpy*cp${PYVER}*macosx*${PYTHON_ARCH}*.whl

# Test installed rawpy
pip install -r dev-requirements.txt "numpy==${NUMPY_VERSION}"
mkdir tmp_for_test
pushd tmp_for_test
python -u -m pytest -v -s ../test
popd

# Exit venv
deactivate

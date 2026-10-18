#!/bin/bash
set -e -x

cd /io

# List python versions
ls /opt/python

# Compute PYBIN from PYTHON_VERSION (e.g., "3.14" -> "cp314-cp314")
PYVER_NO_DOT=${PYTHON_VERSION//./}
PYBIN="/opt/python/cp${PYVER_NO_DOT}-cp${PYVER_NO_DOT}/bin"

if [ ! -d "$PYBIN" ]; then
    echo "Python version $PYTHON_VERSION not found at $PYBIN"
    exit 1
fi
# Source-built test dependencies invoke `python` through /usr/bin/env.
# Keep them on the selected manylinux interpreter, including Python 3.15.
export PATH="$PYBIN:$PATH"
PYVER=${PYTHON_VERSION//.}

# Upgrade pip and prefer binary packages
${PYBIN}/python -m pip install --upgrade pip
export PIP_PREFER_BINARY=1

# Install package and test
${PYBIN}/pip install ./dist/rawpy*cp${PYVER}*manylinux*${PYTHON_ARCH}*.whl
${PYBIN}/pip install -r dev-requirements.txt "numpy==${NUMPY_VERSION}"

mkdir tmp_for_test
pushd tmp_for_test
${PYBIN}/pytest --verbosity=3 -s /io/test
popd

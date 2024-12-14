#!/bin/bash
set -e -x

cd /io

# List python versions
ls /opt/python

if [ $PYTHON_VERSION == "3.9" ]; then
    PYBIN="/opt/python/cp39-cp39/bin"
elif [ $PYTHON_VERSION == "3.10" ]; then
    PYBIN="/opt/python/cp310-cp310/bin"
elif [ $PYTHON_VERSION == "3.11" ]; then
    PYBIN="/opt/python/cp311-cp311/bin"
elif [ $PYTHON_VERSION == "3.12" ]; then
    PYBIN="/opt/python/cp312-cp312/bin"
elif [ $PYTHON_VERSION == "3.13" ]; then
    PYBIN="/opt/python/cp313-cp313/bin"
else
    echo "Unsupported Python version $PYTHON_VERSION"
    exit 1
fi
PYVER=${PYTHON_VERSION//.}

# Install package and test
${PYBIN}/pip install ./dist/rawpy*cp${PYVER}*manylinux*${PYTHON_ARCH}*.whl
${PYBIN}/pip install -r dev-requirements.txt "numpy==${NUMPY_VERSION}"

mkdir tmp_for_test
pushd tmp_for_test
${PYBIN}/pytest --verbosity=3 -s /io/test
popd

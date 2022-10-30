#!/bin/bash
set -e -x

cd /io

# List python versions
ls /opt/python

if [ $PYTHON_VERSION == "3.7" ]; then
    PYBIN="/opt/python/cp37-cp37m/bin"
elif [ $PYTHON_VERSION == "3.8" ]; then
    PYBIN="/opt/python/cp38-cp38/bin"
elif [ $PYTHON_VERSION == "3.9" ]; then
    PYBIN="/opt/python/cp39-cp39/bin"
elif [ $PYTHON_VERSION == "3.10" ]; then
    PYBIN="/opt/python/cp310-cp310/bin"
elif [ $PYTHON_VERSION == "3.11" ]; then
    PYBIN="/opt/python/cp311-cp311/bin"
else
    echo "Unsupported Python version $PYTHON_VERSION"
    exit 1
fi
PYVER=${PYTHON_VERSION//.}

# Install package and test
${PYBIN}/pip install ./dist/rawpy*cp${PYVER}*manylinux*${PYTHON_ARCH}*.whl

${PYBIN}/pip install -r dev-requirements.txt

mkdir tmp_for_test
pushd tmp_for_test
${PYBIN}/pytest --verbosity=3 -s /io/test
popd

#!/bin/bash
set -e -x

# Compile wheels
for PYBIN in ${PYBINS[@]}; do
    ${PYBIN}/pip wheel /io/ -w wheelhouse/
done

ls /io/wheelhouse

# Bundle external shared libraries into the wheels
for whl in wheelhouse/rawpy*.whl; do
    auditwheel repair $whl -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in ${PYBINS[@]}; do
    ${PYBIN}/pip install rawpy --no-index -f /io/wheelhouse
    (cd $HOME; ${PYBIN}/nosetests --verbosity=3 --nocapture /io/test)
done
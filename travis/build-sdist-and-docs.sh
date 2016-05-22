#!/bin/bash
set -e -x

PYBIN=/opt/python/cp35-cp35m/bin

cd /io
$PYBIN/python setup.py sdist
$PYBIN/python setup.py build_ext --inplace
$PYBIN/python setup.py build_sphinx 
ls dist
ls build

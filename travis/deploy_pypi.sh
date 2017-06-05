#!/bin/bash
set -e -x

if [[ "`git log -n 1 --pretty=format:'%d'`" =~ v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    pip install twine
    twine upload -u letmaik -p $PYPI_PASS dist/*
    if [[ "$TRAVIS_OS_NAME" == "linux" ]]; then
        python setup.py build_ext --inplace
        python setup.py build_sphinx
        python setup.py upload_docs
    fi
fi
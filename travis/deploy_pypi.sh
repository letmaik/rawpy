#!/bin/bash
set -e -x

if [[ "$TRAVIS_TAG" =~ v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    pip install twine
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*

    if [[ "$TRAVIS_OS_NAME" == "osx" && "$PYTHON_VERSION" = "3.6" ]]; then
        python setup.py build_ext --inplace
        python setup.py build_sphinx
        # upload via deploy options in .travis.yml
    fi
fi
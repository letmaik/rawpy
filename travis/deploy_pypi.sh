#!/bin/bash
set -e -x

if [[ "$TRAVIS_TAG" =~ v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    pip install twine
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*
fi
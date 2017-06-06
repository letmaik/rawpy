#!/bin/bash
set -e -x

if [[ "`git log -n 1 --pretty=format:'%d'`" =~ v[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    pip install twine
    twine upload -u letmaik -p $PYPI_PASS --skip-existing dist/*
fi
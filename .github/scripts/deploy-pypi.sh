#!/bin/bash
set -e -x

if [[ "$GITHUB_REF" == refs/tags/v* ]]; then
    pip install twine
    twine upload -u $PYPI_USER -p $PYPI_PASS --skip-existing dist/*

    if [[ "$(uname)" == "Darwin" && "$PYTHON_VERSION" = "3.7" ]]; then
        # re-install dependencies as they were removed before running tests (see build-macos.sh)
        brew install jpeg jasper little-cms2
        python setup.py build_ext --inplace
        python setup.py build_sphinx
        # TODO upload via GitHub actions
    fi
fi
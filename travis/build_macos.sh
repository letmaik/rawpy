#!/bin/bash
set -e -x

source travis/travis_retry.sh

export HOMEBREW_NO_BOTTLE_SOURCE_FALLBACK=1
export HOMEBREW_CURL_RETRIES=3
export HOMEBREW_NO_INSTALL_CLEANUP=1

# brew tries to update itself and Ruby during 'brew install ..'' but fails doing so with
# "Homebrew must be run under Ruby 2.3! You're running 2.0.0.".
# Updating brew separately seems to avoid this issue.
travis_retry brew update

# Build wheel
travis_retry git clone https://github.com/MacPython/terryfy.git
set +x # reduce noise
source terryfy/travis_tools.sh
travis_retry get_python_environment macpython $PYTHON_VERSION venv
set -x
travis_retry pip install numpy==$NUMPY_VERSION cython wheel delocate
pip freeze
brew rm --ignore-dependencies jpeg || true
brew install jpeg jasper little-cms2
export CC=clang
export CXX=clang++
export CFLAGS="-arch x86_64"
export CXXFLAGS=$CFLAGS
export LDFLAGS=$CFLAGS
export ARCHFLAGS=$CFLAGS
python setup.py bdist_wheel
delocate-listdeps --all dist/*.whl # lists library dependencies
delocate-wheel --require-archs=x86_64 dist/*.whl # copies library dependencies into wheel
delocate-listdeps --all dist/*.whl # verify

# Install rawpy
pip install dist/*.whl

# Test installed rawpy
travis_retry pip install numpy -U # scipy should trigger an update, but that doesn't happen
travis_retry pip install -r dev-requirements.txt
# make sure it's working without any required libraries installed
brew rm --ignore-dependencies jpeg jasper little-cms2
mkdir tmp_for_test
pushd tmp_for_test
nosetests --verbosity=3 --nocapture ../test
popd

# deploy if git tag
travis/deploy_pypi.sh
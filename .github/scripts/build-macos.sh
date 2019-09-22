#!/bin/bash
set -e -x

source .github/scripts/travis_retry.sh

ls -al /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs || true
ls -al /Developer/SDKs || true

# Used by CMake and clang
export MACOSX_DEPLOYMENT_TARGET=$MACOS_MIN_VERSION

# Use older SDK if 10.6 is targeted, otherwise the build errors:
# "clang: warning: libstdc++ is deprecated; move to libc++ with
#  a minimum deployment target of OS X 10.9 [-Wdeprecated]
#  ld: library not found for -lstdc++""
if [ $MACOS_MIN_VERSION == "10.6" ]; then
    export SDKROOT=/Developer/SDKs/MacOSX10.12.sdk
fi

# Install Python
# Note: The GitHub Actions supplied Python versions are not used
# as they are built without MACOSX_DEPLOYMENT_TARGET/-mmacosx-version-min
# being set to an older target for widest wheel compatibility.
# Instead we install python.org binaries which are built with 10.6/10.9 target
# and hence provide wider compatibility for the wheels we create.
pushd external
git clone https://github.com/matthew-brett/multibuild.git
cd multibuild
source osx_utils.sh
get_macpython_environment $PYTHON_VERSION venv $MACOS_MIN_VERSION
source venv/bin/activate
popd

export HOMEBREW_NO_BOTTLE_SOURCE_FALLBACK=1
export HOMEBREW_CURL_RETRIES=3
export HOMEBREW_NO_INSTALL_CLEANUP=1

# brew tries to update itself and Ruby during 'brew install ..'' but fails doing so with
# "Homebrew must be run under Ruby 2.3! You're running 2.0.0.".
# Updating brew separately seems to avoid this issue.
travis_retry brew update

# Build wheel
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

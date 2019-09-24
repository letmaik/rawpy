#!/bin/bash
set -e -x

source .github/scripts/retry.sh

# General note:
# Apple guarantees forward, but not backward ABI compatibility unless
# the deployment target is set for the oldest supported OS. 
# (https://trac.macports.org/ticket/54332#comment:2)

# Used by CMake, clang, and Python's distutils
export MACOSX_DEPLOYMENT_TARGET=$MACOS_MIN_VERSION

# Work-around issue building on newer XCode versions.
# https://github.com/pandas-dev/pandas/issues/23424#issuecomment-446393981
if [ $MACOS_MIN_VERSION == "10.6" ]; then
    # Note that distutils allows higher but not lower target versions,
    # relative to the target version of Python itself.
    # See https://github.com/python/cpython/blob/9c42f8cda/Lib/distutils/spawn.py#L103-L111.
    # The resulting wheel platform tags still have 10.6 (=target of Python itself),
    # even though technically the wheel should only be run on 10.9 upwards.
    # See https://github.com/pypa/wheel/issues/312.
    export MACOSX_DEPLOYMENT_TARGET=10.9
fi

# Install Python
# Note: The GitHub Actions supplied Python versions are not used
# as they are built without MACOSX_DEPLOYMENT_TARGET/-mmacosx-version-min
# being set to an older target for widest wheel compatibility.
# Instead we install python.org binaries which are built with 10.6/10.9 target
# and hence provide wider compatibility for the wheels we create.
# See https://github.com/actions/setup-python/issues/26.
git clone https://github.com/matthew-brett/multibuild.git
pushd multibuild
set +x # reduce noise
source osx_utils.sh
get_macpython_environment $PYTHON_VERSION venv $MACOS_MIN_VERSION
source venv/bin/activate
set -x
popd

# Install dependencies
retry pip install numpy==$NUMPY_VERSION cython wheel delocate
pip freeze

# Shared library dependencies are built from source to respect MACOSX_DEPLOYMENT_TARGET.
# Bottles from Homebrew cannot be used as they always have a target that
# matches the host OS. Unfortunately, building from source with Homebrew
# is also not an option as the MACOSX_DEPLOYMENT_TARGET env var cannot
# be forwarded to the build (Homebrew cleans the environment).
# See https://discourse.brew.sh/t/it-is-possible-to-build-packages-that-are-compatible-with-older-macos-versions/4421

LIB_INSTALL_PREFIX=$(pwd)/external/libs

# Install libjpeg:
# - pillow (a scikit-image dependency) dependency
# - libraw DNG lossy codec support (requires libjpeg >= 8)
curl --retry 3 http://ijg.org/files/jpegsrc.v9c.tar.gz | tar xz
pushd jpeg-9c
./configure --prefix=$LIB_INSTALL_PREFIX
make install -j
popd

# Install libjasper:
# - libraw RedCine codec support
curl -L --retry 3 https://github.com/mdadams/jasper/archive/version-2.0.16.tar.gz | tar xz
pushd jasper-version-2.0.16
mkdir cmake_build
cd cmake_build
cmake -DCMAKE_INSTALL_PREFIX=$LIB_INSTALL_PREFIX -DCMAKE_BUILD_TYPE=Release \
      -DJAS_ENABLE_OPENGL=OFF -DJAS_ENABLE_DOC=OFF -DJAS_ENABLE_PROGRAMS=OFF \
      -DCMAKE_INSTALL_NAME_DIR=$LIB_INSTALL_PREFIX/lib ..
make install -j
popd

# Install Little CMS 2:
# - libraw lcms support
curl -L --retry 3 https://downloads.sourceforge.net/project/lcms/lcms/2.9/lcms2-2.9.tar.gz | tar xz
pushd lcms2-2.9
# Note: libjpeg and libtiff are only needed for the jpegicc/tifficc tools.
./configure --prefix=$LIB_INSTALL_PREFIX \
            --without-jpeg --without-tiff           
make install -j
popd

ls -al $LIB_INSTALL_PREFIX/lib

export CC=clang
export CXX=clang++
export CFLAGS="-arch x86_64"
export CXXFLAGS=$CFLAGS
export LDFLAGS=$CFLAGS
export ARCHFLAGS=$CFLAGS

# Build wheel
export CMAKE_PREFIX_PATH=$LIB_INSTALL_PREFIX
python setup.py bdist_wheel
delocate-listdeps --all dist/*.whl # lists library dependencies
delocate-wheel --require-archs=x86_64 dist/*.whl # copies library dependencies into wheel
delocate-listdeps --all dist/*.whl # verify

# Dump target versions of dependend libraries.
# Currently, delocate does not support checking those.
# See https://github.com/matthew-brett/delocate/issues/56.
set +x # reduce noise
echo "Dumping LC_VERSION_MIN_MACOSX (pre-10.14) & LC_BUILD_VERSION"
mkdir tmp_wheel
pushd tmp_wheel
unzip ../dist/*.whl
echo rawpy/*.so
otool -l rawpy/*.so | grep -A 3 LC_VERSION_MIN_MACOSX || true
otool -l rawpy/*.so | grep -A 4 LC_BUILD_VERSION || true
for file in rawpy/.dylibs/*.dylib; do
    echo $file
    otool -l $file | grep -A 3 LC_VERSION_MIN_MACOSX || true
    otool -l $file | grep -A 4 LC_BUILD_VERSION || true
done
popd
set -x

# Install rawpy
pip install dist/*.whl

# Test installed rawpy
retry pip install numpy -U # scipy should trigger an update, but that doesn't happen
retry pip install -r dev-requirements.txt
# make sure it's working without any required libraries installed
rm -rf $LIB_INSTALL_PREFIX
mkdir tmp_for_test
pushd tmp_for_test
nosetests --verbosity=3 --nocapture ../test
popd

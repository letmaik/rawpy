#!/bin/bash
set -e -x

CHECK_SHA256=.github/scripts/check_sha256.sh

# General note:
# Apple guarantees forward, but not backward ABI compatibility unless
# the deployment target is set for the oldest supported OS.
# (https://trac.macports.org/ticket/54332#comment:2)

# Used by CMake, clang, and Python's distutils
export MACOSX_DEPLOYMENT_TARGET=$MACOS_MIN_VERSION

# The Python variant to install, see exception below.
export PYTHON_INSTALLER_MACOS_VERSION=$MACOS_MIN_VERSION

# Install Python
# Note: The GitHub Actions supplied Python versions are not used
# as they are built without MACOSX_DEPLOYMENT_TARGET/-mmacosx-version-min
# being set to an older target for widest wheel compatibility.
# Instead we install python.org binaries which are built with 10.6/10.9 target
# and hence provide wider compatibility for the wheels we create.
# See https://github.com/actions/setup-python/issues/26.
git clone https://github.com/multi-build/multibuild.git
pushd multibuild
set +x # reduce noise
source osx_utils.sh
get_macpython_environment $PYTHON_VERSION venv $PYTHON_INSTALLER_MACOS_VERSION
source venv/bin/activate
set -x
popd

# Upgrade pip and prefer binary packages
python -m pip install --upgrade pip
export PIP_PREFER_BINARY=1

# Install dependencies
pip install numpy==$NUMPY_VERSION cython wheel delocate setuptools

# List installed packages
pip freeze

# Shared library dependencies are built from source to respect MACOSX_DEPLOYMENT_TARGET.
# Bottles from Homebrew cannot be used as they always have a target that
# matches the host OS. Unfortunately, building from source with Homebrew
# is also not an option as the MACOSX_DEPLOYMENT_TARGET env var cannot
# be forwarded to the build (Homebrew cleans the environment).
# See https://discourse.brew.sh/t/it-is-possible-to-build-packages-that-are-compatible-with-older-macos-versions/4421

LIB_INSTALL_PREFIX=$(pwd)/external/libs
export CMAKE_PREFIX_PATH=$LIB_INSTALL_PREFIX
export PKG_CONFIG_PATH=$LIB_INSTALL_PREFIX/lib/pkgconfig
export LIBRARY_PATH=$LIB_INSTALL_PREFIX/lib
export PATH=$LIB_INSTALL_PREFIX/bin:$PATH

# Install libjpeg-turbo:
# - pillow (a scikit-image dependency) dependency
# - libjasper dependency
# - libraw DNG lossy codec support (requires libjpeg >= 8)
curl -L --retry 3 -o libjpeg-turbo.tar.gz https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.3/libjpeg-turbo-3.1.3.tar.gz
$CHECK_SHA256 libjpeg-turbo.tar.gz 075920b826834ac4ddf97661cc73491047855859affd671d52079c6867c1c6c0
tar xzf libjpeg-turbo.tar.gz
pushd libjpeg-turbo-3.1.3
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$LIB_INSTALL_PREFIX -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_JPEG8=ON
make install -j
popd

# Install libjasper:
# - libraw RedCine codec support
curl -L --retry 3 -o jasper.tar.gz https://github.com/jasper-software/jasper/archive/version-4.2.5.tar.gz
$CHECK_SHA256 jasper.tar.gz 3f4b1df7cab7a3cc67b9f6e28c730372f030b54b0faa8548a9ee04ae83fffd44
tar xzf jasper.tar.gz
pushd jasper-version-4.2.5
mkdir cmake_build
cd cmake_build
cmake -DCMAKE_INSTALL_PREFIX=$LIB_INSTALL_PREFIX -DCMAKE_BUILD_TYPE=Release \
      -DJAS_ENABLE_OPENGL=OFF -DJAS_ENABLE_DOC=OFF -DJAS_ENABLE_PROGRAMS=OFF \
      -DCMAKE_INSTALL_NAME_DIR=$LIB_INSTALL_PREFIX/lib \
      -DALLOW_IN_SOURCE_BUILD=ON ..
make install -j
popd

# Install Little CMS 2:
# - libraw lcms support
curl -L --retry 3 -o lcms2.tar.gz https://downloads.sourceforge.net/project/lcms/lcms/2.11/lcms2-2.11.tar.gz
$CHECK_SHA256 lcms2.tar.gz dc49b9c8e4d7cdff376040571a722902b682a795bf92985a85b48854c270772e
tar xzf lcms2.tar.gz
pushd lcms2-2.11
# Note: libjpeg and libtiff are only needed for the jpegicc/tifficc tools.
./configure --prefix=$LIB_INSTALL_PREFIX \
            --without-jpeg --without-tiff
make install -j
popd

ls -al $LIB_INSTALL_PREFIX/lib

# By default, wheels are tagged with the architecture of the Python
# installation, which would produce universal2 even if only building
# for x86_64. The following line overrides that behavior.
export _PYTHON_HOST_PLATFORM="macosx-${MACOS_MIN_VERSION}-${PYTHON_ARCH}"

export CC=clang
export CXX=clang++
export CFLAGS="-arch ${PYTHON_ARCH}"
export CXXFLAGS=$CFLAGS
export LDFLAGS=$CFLAGS
export ARCHFLAGS=$CFLAGS

# Build wheel
python setup.py bdist_wheel

delocate-listdeps --all --depending dist/*.whl # lists library dependencies
DYLD_LIBRARY_PATH=$LIB_INSTALL_PREFIX/lib delocate-listdeps --all --depending dist/*.whl # lists library dependencies
DYLD_LIBRARY_PATH=$LIB_INSTALL_PREFIX/lib delocate-wheel --verbose --require-archs=${PYTHON_ARCH} dist/*.whl # copies library dependencies into wheel
delocate-listdeps --all --depending dist/*.whl # verify

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

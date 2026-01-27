#!/bin/bash
set -e -x

cd /io

source .github/scripts/retry.sh

CHECK_SHA256=.github/scripts/check_sha256.sh

# List python versions
ls /opt/python

# Compute PYBIN from PYTHON_VERSION (e.g., "3.14" -> "cp314-cp314")
PYVER_NO_DOT=${PYTHON_VERSION//./}
PYBIN="/opt/python/cp${PYVER_NO_DOT}-cp${PYVER_NO_DOT}/bin"

if [ ! -d "$PYBIN" ]; then
    echo "Python version $PYTHON_VERSION not found at $PYBIN"
    exit 1
fi

# Install zlib:
# - libraw DNG deflate codec support
retry dnf install -y zlib-devel

# Install liblcms2:
# - libraw LCMS support
curl -L --retry 3 -o lcms2.tar.gz https://downloads.sourceforge.net/project/lcms/lcms/2.11/lcms2-2.11.tar.gz
$CHECK_SHA256 lcms2.tar.gz dc49b9c8e4d7cdff376040571a722902b682a795bf92985a85b48854c270772e
tar xzf lcms2.tar.gz
pushd lcms2-2.11
# Note: libjpeg and libtiff are only needed for the jpegicc/tifficc tools.
./configure --prefix=/usr --without-jpeg --without-tiff
make install -j$(nproc)
popd

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
cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_JPEG8=ON
make install -j$(nproc)
popd

# Install libjasper:
# - libraw RedCine codec support
curl -L --retry 3 -o jasper.tar.gz https://github.com/jasper-software/jasper/archive/version-4.2.5.tar.gz
$CHECK_SHA256 jasper.tar.gz 3f4b1df7cab7a3cc67b9f6e28c730372f030b54b0faa8548a9ee04ae83fffd44
tar xzf jasper.tar.gz
pushd jasper-version-4.2.5
mkdir cmake_build
cd cmake_build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release \
      -DJAS_ENABLE_OPENGL=OFF -DJAS_ENABLE_DOC=OFF -DJAS_ENABLE_PROGRAMS=OFF \
      -DALLOW_IN_SOURCE_BUILD=ON ..
make install -j$(nproc)
popd

# Install libraw
libraw_dir=$(pwd)/external/LibRaw
pushd external/LibRaw-cmake
mkdir build
cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DLIBRAW_PATH=$libraw_dir \
    -DENABLE_X3FTOOLS=ON \
    -DENABLE_6BY9RPI=ON \
    -DENABLE_EXAMPLES=OFF \
    -DENABLE_RAWSPEED=OFF \
    -DCMAKE_BUILD_TYPE=Release
make
make install -j$(nproc)
popd

# Install matplotlib (a scikit-image dependency) dependencies
retry dnf install -y libpng-devel freetype-devel

# Install numpy/scipy deps
retry dnf install -y lapack-devel blas-devel

# Upgrade pip and prefer binary packages
${PYBIN}/python -m pip install --upgrade pip
export PIP_PREFER_BINARY=1

# install compile-time dependencies
retry ${PYBIN}/pip install numpy==${NUMPY_VERSION} cython setuptools

# List installed packages
${PYBIN}/pip freeze

# Build rawpy wheel
export LDFLAGS="-Wl,--strip-debug"
${PYBIN}/python setup.py bdist_wheel --dist-dir dist-tmp

# Bundle external shared libraries into wheel and fix the wheel tags
mkdir dist
auditwheel repair dist-tmp/rawpy*.whl -w dist
ls -al dist

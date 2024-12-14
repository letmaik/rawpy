#!/bin/bash
set -e -x

cd /io

source .github/scripts/retry.sh

CHECK_SHA256=.github/scripts/check_sha256.sh

# List python versions
ls /opt/python

if [ $PYTHON_VERSION == "3.9" ]; then
    PYBIN="/opt/python/cp39-cp39/bin"
elif [ $PYTHON_VERSION == "3.10" ]; then
    PYBIN="/opt/python/cp310-cp310/bin"
elif [ $PYTHON_VERSION == "3.11" ]; then
    PYBIN="/opt/python/cp311-cp311/bin"
elif [ $PYTHON_VERSION == "3.12" ]; then
    PYBIN="/opt/python/cp312-cp312/bin"
elif [ $PYTHON_VERSION == "3.13" ]; then
    PYBIN="/opt/python/cp313-cp313/bin"
else
    echo "Unsupported Python version $PYTHON_VERSION"
    exit 1
fi

# Install zlib:
# - libraw DNG deflate codec support
retry yum install -y zlib-devel

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

# Install libjpeg:
# - pillow (a scikit-image dependency) dependency
# - libjasper dependency
# - libraw DNG lossy codec support (requires libjpeg >= 8)
# TODO: switch to libjpeg-turbo
curl --retry 3 -o jpegsrc.tar.gz http://ijg.org/files/jpegsrc.v9d.tar.gz
$CHECK_SHA256 jpegsrc.tar.gz 2303a6acfb6cc533e0e86e8a9d29f7e6079e118b9de3f96e07a71a11c082fa6a
tar xzf jpegsrc.tar.gz
pushd jpeg-9d
./configure --prefix=/usr
make install -j$(nproc)
popd

# Install libjasper:
# - libraw RedCine codec support
curl -L --retry 3 -o jasper.tar.gz https://github.com/jasper-software/jasper/archive/version-2.0.32.tar.gz
$CHECK_SHA256 jasper.tar.gz a3583a06698a6d6106f2fc413aa42d65d86bedf9a988d60e5cfa38bf72bc64b9
tar xzf jasper.tar.gz
pushd jasper-version-2.0.32
mkdir cmake_build
cd cmake_build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release \
      -DJAS_ENABLE_OPENGL=OFF -DJAS_ENABLE_DOC=OFF -DJAS_ENABLE_PROGRAMS=OFF ..
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
retry yum install -y libpng-devel freetype-devel

# Install numpy/scipy deps
retry yum install -y lapack-devel blas-devel

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

#!/bin/bash
set -e -x

cd /io

source .github/scripts/retry.sh

CHECK_SHA256=.github/scripts/check_sha256.sh

# List python versions
ls /opt/python

if [ $PYTHON_VERSION == "3.6" ]; then
    PYBIN="/opt/python/cp36-cp36m/bin"
elif [ $PYTHON_VERSION == "3.7" ]; then
    PYBIN="/opt/python/cp37-cp37m/bin"
elif [ $PYTHON_VERSION == "3.8" ]; then
    PYBIN="/opt/python/cp38-cp38/bin"
elif [ $PYTHON_VERSION == "3.9" ]; then
    PYBIN="/opt/python/cp39-cp39/bin"
elif [ $PYTHON_VERSION == "3.10" ]; then
    PYBIN="/opt/python/cp310-cp310/bin"
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
$CHECK_SHA256 jpegsrc.tar.gz 6c434a3be59f8f62425b2e3c077e785c9ce30ee5874ea1c270e843f273ba71ee
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
    -DLIBRAW_PATH=$libraw_dir \
    -DENABLE_X3FTOOLS=ON \
    -DENABLE_6BY9RPI=ON \
    -DENABLE_EXAMPLES=OFF \
    -DENABLE_RAWSPEED=OFF \
    -DCMAKE_BUILD_TYPE=Release
make
make install -j$(nproc)
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/99local.conf
ldconfig
popd

# Install matplotlib (a scikit-image dependency) dependencies
retry yum install -y libpng-devel freetype-devel

# Install numpy/scipy deps
retry yum install -y lapack-devel blas-devel

# install compile-time dependencies
retry ${PYBIN}/pip install numpy==${NUMPY_VERSION} cython

# List installed packages
${PYBIN}/pip freeze

# Build rawpy wheel
rm -rf wheelhouse
retry ${PYBIN}/pip wheel . -w wheelhouse

# Bundle external shared libraries into wheel
auditwheel repair wheelhouse/rawpy*.whl -w wheelhouse

# Install package and test
${PYBIN}/pip install rawpy --no-index -f wheelhouse

retry ${PYBIN}/pip install -r dev-requirements.txt
retry ${PYBIN}/pip install -U numpy # scipy should trigger an update, but that doesn't happen

pushd $HOME
${PYBIN}/nosetests --verbosity=3 --nocapture /io/test
popd

# Move wheel to dist/ folder for easier deployment
mkdir -p dist
mv wheelhouse/rawpy*manylinux*.whl dist/

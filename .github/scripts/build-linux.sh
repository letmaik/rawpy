#!/bin/bash
set -e -x

bash --version

cd /io

source .github/scripts/retry.sh

# List python versions
ls /opt/python

if [ $PYTHON_VERSION == "3.5" ]; then
    PYBIN="/opt/python/cp35-cp35m/bin"
elif [ $PYTHON_VERSION == "3.6" ]; then
    PYBIN="/opt/python/cp36-cp36m/bin"
elif [ $PYTHON_VERSION == "3.7" ]; then
    PYBIN="/opt/python/cp37-cp37m/bin"
elif [ $PYTHON_VERSION == "3.8" ]; then
    PYBIN="/opt/python/cp38-cp38/bin"
else
    echo "Unsupported Python version $PYTHON_VERSION"
    exit 1
fi

# Install build tools
retry yum install -y cmake

# Install zlib:
# - libraw DNG deflate codec support
retry yum install -y zlib-devel

# Install liblcms2:
# - libraw LCMS support
retry yum install -y lcms2-devel

# Install libjpeg:
# - pillow (a scikit-image dependency) dependency
# - libraw DNG lossy codec support (requires libjpeg >= 8)
# CentOS 6 has libjpeg 6 only, so build from source.
curl --retry 3 http://ijg.org/files/jpegsrc.v9d.tar.gz | tar xz
pushd jpeg-9d
./configure --prefix=/usr
make install -j$(nproc)
popd

# Install libjasper:
# - libraw RedCine codec support
# CentOS 6 has libjasper, but since it depends on libjpeg we'll build from
# source, otherwise we would use two different libjpeg versions.
curl -L --retry 3 https://github.com/mdadams/jasper/archive/version-2.0.16.tar.gz | tar xz
pushd jasper-version-2.0.16
mkdir cmake_build
cd cmake_build
cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release \
      -DJAS_ENABLE_OPENGL=OFF -DJAS_ENABLE_DOC=OFF -DJAS_ENABLE_PROGRAMS=OFF ..
make install -j$(nproc)
popd

# Install libraw
pushd external
cp -R LibRaw-cmake/* LibRaw
pushd LibRaw
cmake -DENABLE_EXAMPLES=OFF -DENABLE_RAWSPEED=OFF -DCMAKE_BUILD_TYPE=Release .
make
make install -j$(nproc)
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/99local.conf
ldconfig
popd
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
mv wheelhouse/rawpy*manylinux2010*.whl dist/

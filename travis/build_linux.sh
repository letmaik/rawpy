#!/bin/bash
set -e -x

cd /io

source travis/travis_retry.sh

# List python versions
ls /opt/python

PYBINS=(
  "/opt/python/cp27-cp27mu/bin"
  "/opt/python/cp34-cp34m/bin"
  "/opt/python/cp35-cp35m/bin"
  "/opt/python/cp36-cp36m/bin"
  )

# Install build tools
travis_retry yum install -y cmake

# Install libraw
pushd external
cp -R LibRaw-cmake/* LibRaw
pushd LibRaw
cmake . -DENABLE_EXAMPLES=OFF -DENABLE_RAWSPEED=OFF
cmake --build . --target install
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/99local.conf
ldconfig
popd
popd

# Install matplotlib (a scikit-image dependency) dependencies
travis_retry yum install -y libpng-devel freetype-devel

# Install pillow (a scikit-image dependency) dependencies
travis_retry yum install -y libjpeg-devel

# Install numpy/scipy deps
travis_retry yum install -y lapack-devel blas-devel

# Build rawpy wheels
for PYBIN in ${PYBINS[@]}; do
    case ${PYBIN} in
        *27*) NUMPY_VERSION="1.7.2";;
        *34*) NUMPY_VERSION="1.8.*";;
        *35*) NUMPY_VERSION="1.9.*";;
        *36*) NUMPY_VERSION="1.11.*";;
    esac

    ${PYBIN}/pip install numpy==${NUMPY_VERSION} cython wheel

    ${PYBIN}/python setup.py bdist_wheel -d wheelhouse
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair $whl -w wheelhouse
done

# Build sdist
${PYBINS[0]}/python setup.py sdist

# Install packages and test
for PYBIN in ${PYBINS[@]}; do
    ${PYBIN}/pip install rawpy --no-index -f wheelhouse
    
    ${PYBIN}/pip install -r dev-requirements.txt
    (cd $HOME; ${PYBIN}/nosetests --verbosity=3 --nocapture /io/test)
done

# Build docs
${PYBINS[0]}/python setup.py build_ext --inplace
${PYBINS[0]}/python setup.py build_sphinx

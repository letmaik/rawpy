#!/bin/bash
set -e -x

source travis/travis_retry.sh

# List python versions
ls /opt/python

PYBINS=(
  "/opt/python/cp27-cp27mu/bin"
  "/opt/python/cp34-cp34m/bin"
  "/opt/python/cp35-cp35m/bin"
  "/opt/python/cp36-cp36m/bin"
  )

declare -A NUMPY_VERSION
NUMPY_VERSION=(
  ["/opt/python/cp27-cp27mu/bin"]="1.7.2"
  ["/opt/python/cp34-cp34m/bin"]="1.8.*"
  ["/opt/python/cp35-cp35m/bin"]="1.9.*"
  ["/opt/python/cp36-cp36m/bin"]="1.11.*"
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
    ${PYBIN}/pip install numpy==${NUMPY_VERSION[${PYBIN}]}

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

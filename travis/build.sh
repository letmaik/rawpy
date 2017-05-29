#!/bin/bash
set -e -x

# https://github.com/travis-ci/travis-build/blob/master/lib/travis/build/templates/header.sh
travis_retry() {
  local result=0
  local count=1
  while [ $count -le 3 ]; do
    [ $result -ne 0 ] && {
      echo -e "\n${ANSI_RED}The command \"$@\" failed. Retrying, $count of 3.${ANSI_RESET}\n" >&2
    }
    "$@"
    result=$?
    [ $result -eq 0 ] && break
    count=$(($count + 1))
    sleep 1
  done

  [ $count -gt 3 ] && {
    echo -e "\n${ANSI_RED}The command \"$@\" failed 3 times.${ANSI_RESET}\n" >&2
  }

  return $result
}

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
cd /opt
travis_retry git clone $LIBRAW_GIT
travis_retry git clone $LIBRAW_CMAKE_GIT
cp -R LibRaw-cmake/* LibRaw
cd LibRaw
git checkout `git describe --abbrev=0 --tags`
cmake . -DENABLE_EXAMPLES=OFF -DENABLE_RAWSPEED=OFF
make
make install
echo "/usr/local/lib" | tee /etc/ld.so.conf.d/99local.conf
ldconfig
cd /io

# Install matplotlib (a scikit-image dependency) dependencies
travis_retry yum install -y libpng-devel freetype-devel

# Install pillow (a scikit-image dependency) dependencies
travis_retry yum install -y libjpeg-devel

# Install numpy/scipy deps
travis_retry yum install -y lapack-devel blas-devel

# Install python dependencies
for PYBIN in ${PYBINS[@]}; do
    # numpy 1.7 is our minimum supported version
    ${PYBIN}/pip install numpy==1.7.2

    # the following versions still support numpy 1.7
    ${PYBIN}/pip install scipy==0.18.1    
    ${PYBIN}/pip install matplotlib==2.0.2
    ${PYBIN}/pip install scikit-image==0.12.3
        
    ${PYBIN}/pip install -r /io/dev-requirements.txt
    ${PYBIN}/pip freeze
done

# Build rawpy distribution
source /io/travis/build-sdist-and-docs.sh
source /io/travis/build-wheels.sh

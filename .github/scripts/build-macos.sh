#!/bin/bash
set -e -x

CHECK_SHA256=.github/scripts/check_sha256.sh

# General note:
# Apple guarantees forward, but not backward ABI compatibility unless
# the deployment target is set for the oldest supported OS.
# (https://trac.macports.org/ticket/54332#comment:2)

# Used by CMake, clang, and Python's distutils
export MACOSX_DEPLOYMENT_TARGET=$MACOS_MIN_VERSION

# Python is selected by actions/setup-python in the workflow.
python -m venv venv
source venv/bin/activate

# Upgrade pip and prefer binary packages
python -m pip install --upgrade pip
export PIP_PREFER_BINARY=1

# Install delocate for bundling shared libraries into the wheel
pip install delocate

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
pip wheel . --wheel-dir dist --no-deps

DYLD_LIBRARY_PATH=$LIB_INSTALL_PREFIX/lib delocate-listdeps --all --depending dist/*.whl # lists library dependencies
DYLD_LIBRARY_PATH=$LIB_INSTALL_PREFIX/lib delocate-wheel --verbose --require-archs=${PYTHON_ARCH} dist/*.whl # copies library dependencies into wheel
delocate-listdeps --all --depending dist/*.whl # verify

# Verify the minimum macOS version of every Mach-O binary in the wheel.
set +x # reduce noise
echo "Checking Mach-O deployment targets (requested: $MACOS_MIN_VERSION)"
mkdir tmp_wheel
pushd tmp_wheel
unzip ../dist/*.whl

check_minos() {
    local file=$1
    local load_commands
    load_commands=$(otool -l "$file")
    printf '%s\n' "$load_commands" | awk -v file="$file" -v expected="$MACOS_MIN_VERSION" '
        function normalize(version, parts, count) {
            count = split(version, parts, /\./)
            if (count == 3 && parts[3] == 0) {
                return parts[1] "." parts[2]
            }
            return version
        }
        function finish_command() {
            if (command != "") {
                command_count++
                if (!found_version) {
                    printf "ERROR: %s has no deployment target in %s\n", file, command > "/dev/stderr"
                    failed = 1
                }
            }
        }
        $1 == "cmd" {
            finish_command()
            command = ""
            found_version = 0
            if ($2 == "LC_BUILD_VERSION") {
                command = "minos"
            } else if ($2 == "LC_VERSION_MIN_MACOSX") {
                command = "version"
            }
            next
        }
        command == "minos" && $1 == "minos" ||
        command == "version" && $1 == "version" {
            found_version = 1
            printf "%s: %s %s\n", file, $1, $2
            if (normalize($2) != normalize(expected)) {
                printf "ERROR: %s targets macOS %s, expected %s\n", file, $2, expected > "/dev/stderr"
                failed = 1
            }
        }
        END {
            finish_command()
            if (command_count == 0) {
                printf "ERROR: %s has no macOS deployment target load command\n", file > "/dev/stderr"
                failed = 1
            }
            exit failed
        }
    '
}

shopt -s nullglob
binaries=(rawpy/*.so rawpy/*.dylib rawpy/.dylibs/*.dylib)
if ((${#binaries[@]} == 0)); then
    echo "ERROR: no Mach-O binaries found in wheel" >&2
    exit 1
fi
for file in "${binaries[@]}"; do
    check_minos "$file"
done
popd
set -x

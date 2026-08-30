#!/bin/bash
set -euxo pipefail

python -m pip install --upgrade pip
brew install libraw

# Put Homebrew's LibRaw before setuptools' library_dirs. The bundled build
# must still link against the LibRaw that it just compiled.
libraw_prefix=$(brew --prefix libraw)
export LDFLAGS="-L${libraw_prefix}/lib"

python -m pip wheel . --wheel-dir dist --no-deps

wheel=$(find dist -name 'rawpy-*.whl' -print -quit)
test -n "${wheel}"

wheel_dir=tmp_macos_libraw_linking
rm -rf "${wheel_dir}"
unzip -q "${wheel}" -d "${wheel_dir}"

extensions=("${wheel_dir}"/rawpy/_rawpy*.so)
test "${#extensions[@]}" -eq 1

dependencies=$(otool -L "${extensions[0]}")
echo "${dependencies}"

if grep -Fq "${libraw_prefix}/lib/libraw_r" <<<"${dependencies}"; then
    echo "ERROR: rawpy linked against Homebrew LibRaw instead of the bundled library"
    exit 1
fi

grep -Eq '[[:space:]]@rpath/libraw_r\.[0-9]+\.dylib' <<<"${dependencies}"

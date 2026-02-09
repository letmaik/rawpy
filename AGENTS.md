# Agent Development Guide

This repository wraps the C++ `LibRaw` library using Cython.

## Prerequisites

Before starting, ensure you have:
- **Python 3.9+**
- **C++ compiler** - `apt install g++` (Ubuntu) / Xcode Command Line Tools (macOS)

## Critical: Compilation Required

**You are working with Cython (`.pyx`) files.**
Changes to `rawpy/_rawpy.pyx` or C++ files **will not take effect** until you recompile.

| File type | After editing... |
|-----------|------------------|
| `.py` files | Changes apply immediately (editable install) |
| `.pyx` files | Must run `bash scripts/rebuild.sh` |
| C++ files in `external/` | Must run `bash scripts/rebuild.sh` |
| `MANIFEST.in` | Rebuild: `bash scripts/build_dist.sh` |

## Quick Commands

| Task | Command |
|------|---------|
| First-time setup | `bash scripts/setup_dev_env.sh` |
| Setup with specific Python | `bash scripts/setup_dev_env.sh 3.12` |
| Activate environment | `source .venv/bin/activate` |
| Rebuild after .pyx/C++ changes | `bash scripts/rebuild.sh` |
| Quick sanity check | `bash scripts/dev_check.sh` |
| Build sdist + wheel | `bash scripts/build_dist.sh` |
| Test built sdist | `bash scripts/test_dist.sh sdist` |
| Test built wheel | `bash scripts/test_dist.sh wheel` |
| Test with numpy version | `bash scripts/test_dist.sh wheel 2.0.2` |
| Test sdist with system libraw | `bash scripts/build_dist.sh && RAWPY_USE_SYSTEM_LIBRAW=1 bash scripts/test_dist.sh sdist` |
| Test wheel with system libraw | `RAWPY_USE_SYSTEM_LIBRAW=1 bash scripts/build_dist.sh && RAWPY_USE_SYSTEM_LIBRAW=1 bash scripts/test_dist.sh wheel` |
| Run single test | `pytest test/test_basic.py::testName -v` |
| Run all tests | `pytest test/` |
| Type check | `mypy rawpy` |
| Switch numpy version | `bash scripts/setup_numpy.sh 2.0.2` |

> **System libraw requires LibRaw ≥ 0.21.** Ubuntu 22.04's `libraw-dev` (0.20.2) is
> too old. Use Ubuntu 24.04+ or build without `RAWPY_USE_SYSTEM_LIBRAW`.
>
> Note: The sdist build command does **not** use `RAWPY_USE_SYSTEM_LIBRAW=1`
> because sdist just packages source files — it doesn't compile anything. The
> env var is only needed at install/test time, when pip builds the sdist from
> source. For wheel, the env var is needed at both build **and** test time.

## Environment Setup

**First time only:**
```bash
bash scripts/setup_dev_env.sh
```

This will:
1. Create a `.venv` virtual environment
2. Check for required system dependencies (cmake, C++ compiler)
3. Initialize git submodules (LibRaw source)
4. Install Python dependencies
5. Build and install rawpy in editable mode

**With a specific Python version (Ubuntu only):**
```bash
bash scripts/setup_dev_env.sh 3.12
```

This installs the requested Python via the deadsnakes PPA, creates a `.venv`
with it, then runs the full setup. You can also use `scripts/setup_python.sh`
directly if you only need to switch the Python version without rebuilding.

**For subsequent sessions:**
```bash
source .venv/bin/activate
```

## Architecture

| Path | Purpose |
|------|---------|
| `rawpy/_rawpy.pyx` | Main Cython implementation (RawPy class, C++ bindings) |
| `rawpy/_rawpy.cpp` | **Generated** C++ from `.pyx` — do not edit manually. `setup.py` calls `cythonize()` which regenerates this, but only when the `.pyx` has a newer timestamp than the `.cpp`. A stale `.cpp` from a previous build can cause failures if the NumPy ABI has changed. `scripts/rebuild.sh` deletes it to force regeneration. |
| `rawpy/_rawpy.pyi` | Type stubs (update when changing API) |
| `rawpy/__init__.py` | Python entry point |
| `rawpy/enhance.py` | Pure Python utilities (bad pixel repair, etc.) |
| `external/LibRaw/` | LibRaw C++ library (git submodule) |
| `external/LibRaw/libraw/*.h` | LibRaw headers (check these for C++ signatures) |
| `external/LibRaw-cmake/` | CMake build system for LibRaw (git submodule) |
| `setup.py` | Build configuration (compiles LibRaw from source, links Cython extension) |
| `tmp/` | Scratch directory for build logs etc. (git-ignored) |
| `.github/workflows/ci.yml` | CI workflow (build matrix for Linux/macOS/Windows × Python versions) |
| `.github/scripts/` | Platform-specific CI build/test scripts |

## Common Tasks

### Adding a new LibRaw method

1. Find the C++ signature in `external/LibRaw/libraw/libraw.h`
2. Add the `cdef extern` declaration in `rawpy/_rawpy.pyx`
3. Add a Python method in the `RawPy` class in `rawpy/_rawpy.pyx`
4. Add type stub in `rawpy/_rawpy.pyi`
5. Rebuild: `bash scripts/rebuild.sh`
6. Add a test in `test/`

### Testing sdist and wheel artifacts

The editable install (`pip install -e .`) is convenient for development but
doesn't catch packaging problems (missing files in `MANIFEST.in`, broken
build isolation, etc.). To test what end-users will get:

```bash
# Build sdist and wheel (output in dist/)
bash scripts/build_dist.sh

# Test the sdist — builds from source in a clean venv, then runs pytest
bash scripts/test_dist.sh sdist

# Test the wheel
bash scripts/test_dist.sh wheel

# Test with a specific numpy version
bash scripts/test_dist.sh sdist 2.0.2

# Test with a specific Python version (Ubuntu, via deadsnakes)
bash scripts/setup_python.sh 3.12
bash scripts/build_dist.sh
bash scripts/test_dist.sh sdist
```

The test script creates an isolated `.venv-test` (separate from the dev
`.venv`), installs the artifact, runs the test suite from a temp directory
(so the source tree's `rawpy/` isn't accidentally imported), and cleans up
automatically.

**Tip:** Building from source (sdist install, `pip install .`, etc.) compiles
LibRaw and the Cython extension, which can take several minutes. Use `tee` to
save output to `tmp/` (git-ignored) while still seeing progress:

```bash
mkdir -p tmp
bash scripts/build_dist.sh 2>&1 | tee tmp/build.log
# Then inspect:
grep -i error tmp/build.log  # just errors
tail -30 tmp/build.log       # last 30 lines
```

> `tee` overwrites by default (like `>`), so re-running always gives a fresh log.

### Running specific tests

```bash
# Run a single test
pytest test/test_basic.py::testFileOpenAndPostProcess -v

# Run tests matching a pattern
pytest -k "thumbnail" -v

# Run with print output visible
pytest -s test/test_basic.py
```

## Troubleshooting

### "No module named rawpy._rawpy"
The Cython extension isn't built. Run:
```bash
bash scripts/rebuild.sh
```

### "PyArray_Descr has no member named 'subarray'" or similar NumPy ABI errors
The generated `_rawpy.cpp` is stale (compiled against a different NumPy version).
`scripts/rebuild.sh` already handles this by deleting the `.cpp` so `cythonize()`
regenerates it. Just re-run `bash scripts/rebuild.sh`. To fix manually:
```bash
rm rawpy/_rawpy.cpp
pip install --no-build-isolation -e .
```

### "cmake: command not found"
cmake is installed automatically as a build dependency via `pyproject.toml`.
If you see this error during an editable install (`--no-build-isolation`),
install it into your venv:
```bash
pip install cmake
```

### "fatal error: libraw/libraw.h: No such file or directory"
Git submodules aren't initialized:
```bash
git submodule update --init --recursive
```

### Build fails with compiler errors
Ensure you have a C++ compiler:
```bash
# Ubuntu/Debian
sudo apt install g++

# macOS (installs clang)
xcode-select --install
```

### Mypy errors about missing stubs
If you added new API, update `rawpy/_rawpy.pyi` to match.

### System libraw build fails with missing struct members
The system `libraw-dev` is too old. rawpy requires LibRaw ≥ 0.21.
Ubuntu 22.04 ships LibRaw 0.20.2, which is incompatible. Errors look like:
```
error: 'libraw_raw_unpack_params_t' was not declared in this scope
error: 'struct libraw_image_sizes_t' has no member named 'raw_inset_crops'
```
Use Ubuntu 24.04+ (ships 0.22) or build without `RAWPY_USE_SYSTEM_LIBRAW=1`
(the default), which compiles LibRaw 0.22 from the bundled submodule.

## CI Architecture

The CI workflow is in `.github/workflows/ci.yml`. It builds wheels across a matrix:

| Platform | Runner | Container | Architectures |
|----------|--------|-----------|---------------|
| Linux | `ubuntu-latest` | `manylinux_2_28` (RHEL-based) | x86_64, aarch64 (via QEMU) |
| macOS | `macos-15` | native | arm64 |
| Windows | `windows-2022` | native | x86_64 |

Build scripts in `.github/scripts/`:
- `build-linux.sh` — runs inside Docker; installs deps, builds wheel, runs `auditwheel`
- `build-macos.sh` — installs deps from source (respects `MACOSX_DEPLOYMENT_TARGET`), uses `delocate`
- `build-windows.ps1` — uses vcpkg for deps, VS build tools

### Reproducing CI builds locally

```bash
# Build sdist + wheel with build isolation (recommended):
bash scripts/build_dist.sh

# Low-level alternative (what CI does for wheels):
pip wheel . --wheel-dir dist --no-deps

# Build without isolation (faster, for local dev):
pip install --no-build-isolation -e .
```

Note: `python -m build` (used by `build_dist.sh`) and `pip wheel .` both use
build isolation and create a fresh environment from `pyproject.toml`'s
`build-system.requires`. This is different from the local dev workflow
(`--no-build-isolation`) which reuses the current venv.

### Reproducing CI test failures locally

CI tests run across multiple Python and NumPy versions. Type checking (mypy)
is particularly sensitive to the NumPy stubs version bundled with each NumPy
release.

**Test artifacts against a specific Python + NumPy (closest to CI):**
```bash
# Install Python 3.12 via deadsnakes, build artifacts, and test sdist
bash scripts/setup_python.sh 3.12
bash scripts/build_dist.sh
bash scripts/test_dist.sh sdist 2.0.2
```

**Test with a specific NumPy version (editable install):**
```bash
# Switch to numpy 2.0.x, then use normal commands
bash scripts/setup_numpy.sh 2.0.2
source .venv/bin/activate
pytest test/test_mypy.py -v

# Switch back when done
bash scripts/setup_numpy.sh 2.2.6
```

**Test with a specific Python version (editable install):**
```bash
# Install Python 3.12 and rebuild everything
bash scripts/setup_dev_env.sh 3.12

# Then run tests
source .venv/bin/activate
pytest test/ -v
```

**Reference CI NumPy versions** (check `.github/workflows/ci.yml` test matrix):

| Python | NumPy |
|--------|-------|
| 3.9–3.12 | 2.0.* |
| 3.13 | 2.1.* |
| 3.14 | 2.4.* |

## Platform-Specific Notes

- **Linux (RHEL/manylinux):** CMake's `GNUInstallDirs` installs libraries to
  `lib64/` instead of `lib/` on 64-bit RHEL-based systems. The `setup.py`
  handles this by passing `-DCMAKE_INSTALL_LIBDIR=lib` to cmake. If you modify
  cmake arguments in `setup.py`, always keep this flag.
- **macOS:** `-DCMAKE_INSTALL_NAME_DIR` is required for dylib resolution.
  Build scripts install dependencies from source to control `MACOSX_DEPLOYMENT_TARGET`.
- **Windows:** Uses a separate `windows_libraw_compile()` code path with
  NMake Makefiles generator and vcpkg for native dependencies.

## Examples

See `examples/` for API usage:
- `basic_process.py` - Load RAW, postprocess to RGB, save
- `thumbnail_extract.py` - Extract embedded JPEG thumbnail
- `bad_pixel_repair.py` - Detect and repair bad pixels

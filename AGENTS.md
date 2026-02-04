# Agent Development Guide

This repository wraps the C++ `LibRaw` library using Cython.

## Prerequisites

Before starting, ensure you have:
- **Python 3.9+**
- **cmake** - `apt install cmake` (Ubuntu) / `brew install cmake` (macOS)
- **C++ compiler** - `apt install g++` (Ubuntu) / Xcode Command Line Tools (macOS)

## Critical: Compilation Required

**You are working with Cython (`.pyx`) files.**
Changes to `rawpy/_rawpy.pyx` or C++ files **will not take effect** until you recompile.

| File type | After editing... |
|-----------|------------------|
| `.py` files | Changes apply immediately (editable install) |
| `.pyx` files | Must run `bash scripts/rebuild.sh` |
| C++ files in `external/` | Must run `bash scripts/rebuild.sh` |

## Quick Commands

| Task | Command |
|------|---------|
| First-time setup | `bash scripts/setup_agent_env.sh` |
| Activate environment | `source .venv/bin/activate` |
| Rebuild after .pyx/C++ changes | `bash scripts/rebuild.sh` |
| Quick sanity check | `bash scripts/agent_check.sh` |
| Run single test | `pytest test/test_basic.py::testName -v` |
| Run all tests | `pytest test/` |
| Type check | `mypy rawpy` |

## Environment Setup

**First time only:**
```bash
bash scripts/setup_agent_env.sh
```

This will:
1. Create a `.venv` virtual environment
2. Check for required system dependencies (cmake, C++ compiler)
3. Initialize git submodules (LibRaw source)
4. Install Python dependencies
5. Build and install rawpy in editable mode

**For subsequent sessions:**
```bash
source .venv/bin/activate
```

## Architecture

| Path | Purpose |
|------|---------|
| `rawpy/_rawpy.pyx` | Main Cython implementation (RawPy class, C++ bindings) |
| `rawpy/_rawpy.pyi` | Type stubs (update when changing API) |
| `rawpy/__init__.py` | Python entry point |
| `rawpy/enhance.py` | Pure Python utilities (bad pixel repair, etc.) |
| `external/LibRaw/` | LibRaw C++ library (git submodule) |
| `external/LibRaw/libraw/*.h` | LibRaw headers (check these for C++ signatures) |

## Common Tasks

### Adding a new LibRaw method

1. Find the C++ signature in `external/LibRaw/libraw/libraw.h`
2. Add the `cdef extern` declaration in `rawpy/_rawpy.pyx`
3. Add a Python method in the `RawPy` class in `rawpy/_rawpy.pyx`
4. Add type stub in `rawpy/_rawpy.pyi`
5. Rebuild: `bash scripts/rebuild.sh`
6. Add a test in `test/`

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

### "cmake: command not found"
Install cmake via your system package manager:
```bash
# Ubuntu/Debian
sudo apt install cmake

# macOS
brew install cmake

# Fedora
sudo dnf install cmake
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

## Examples

See `examples/` for API usage:
- `basic_process.py` - Load RAW, postprocess to RGB, save
- `thumbnail_extract.py` - Extract embedded JPEG thumbnail
- `bad_pixel_repair.py` - Detect and repair bad pixels

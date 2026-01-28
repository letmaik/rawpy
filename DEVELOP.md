# Development Guide

This document provides guidance on how to get started with rawpy development and how to perform releases.

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Git**: For cloning the repository and submodules
- **C/C++ Compiler**: 
  - Linux: GCC
  - macOS: Xcode Command Line Tools
  - Windows: Visual Studio 2017 or higher

### Platform-Specific Requirements

#### Linux
- `libraw-dev` (optional, can also build from source via git submodules)
- `pkg-config`
- `cmake` (if building LibRaw from source)

#### macOS
- Xcode Command Line Tools
- LibRaw is built automatically as part of the build process

#### Windows
- Visual Studio 2017 or higher
- CMake is downloaded automatically during build

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/letmaik/rawpy.git
cd rawpy
git submodule update --init
```

The `git submodule update --init` command clones the LibRaw library and related submodules into the `external/` directory.

### 2. Install Python Dependencies

```bash
pip install -r dev-requirements.txt
```

This installs:
- Build dependencies: `wheel`, `cython`, `delocate` (macOS)
- Runtime dependencies: `numpy`
- Optional dependencies: `scikit-image`
- Test dependencies: `pytest`, `imageio`, `setuptools`
- Documentation dependencies: `sphinx`, `sphinx_rtd_theme`

### 3. Build and Install in Development Mode

#### Using System LibRaw (Linux only)

On Ubuntu/Debian:
```bash
sudo apt-get install libraw-dev
pip install -e .
```

#### Building LibRaw from Source (Recommended for macOS/Windows)

```bash
pip install -e .
```

This will automatically:
- Build LibRaw from the git submodules
- Compile the Cython extension
- Install rawpy in editable mode

**Note**: On Linux, you may need to update the library cache:
```bash
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/99local.conf
sudo ldconfig
```

### 4. Verify Installation

```bash
python -c "import rawpy; print(rawpy.__version__); print(rawpy.libraw_version)"
```

## Building

### Build Wheels

The project uses GitHub Actions for building wheels across multiple platforms. To build locally:

#### Linux
```bash
# Inside manylinux container
.github/scripts/build-linux.sh
```

#### macOS
```bash
.github/scripts/build-macos.sh
```

#### Windows
```powershell
.github/scripts/build-windows.ps1
```

## Testing

### Run Tests

```bash
pytest test/
```

### Run Specific Test

```bash
pytest test/test_basic.py
```

### Test Wheel Installation

After building wheels:

#### Linux
```bash
.github/scripts/test-linux.sh
```

#### macOS
```bash
.github/scripts/test-macos.sh
```

#### Windows
```powershell
.github/scripts/test-windows.ps1
```

## Documentation

### Build Documentation

```bash
sphinx-build -b html docs dist-docs
```

The generated HTML documentation will be in `dist-docs/`.

### View Documentation Locally

```bash
python -m http.server --directory dist-docs 8000
```

Then open http://localhost:8000 in your browser.

## Code Structure

- `rawpy/` - Python package source code
  - `_rawpy.pyx` - Cython wrapper around LibRaw
  - `_version.py` - Version information
  - `enhance.py` - Bad pixel detection and repair functionality
- `external/` - Git submodules for LibRaw and related libraries
- `test/` - Test files and test data
- `docs/` - Sphinx documentation source
- `.github/` - GitHub Actions workflows and build scripts
  - `workflows/ci.yml` - Main CI/CD workflow
  - `scripts/` - Platform-specific build and test scripts

## Release Process

### Release Types

- **Stable releases** (e.g., `0.25.0`): Built against a stable LibRaw release
- **Pre-releases** (e.g., `0.25.0a1`, `0.25.0b1`): Built against recent LibRaw snapshots

### Creating a Stable Release

1. **Update Version Number**
   
   Edit `rawpy/_version.py`:
   ```python
   __version__ = "X.Y.Z"  # e.g., "0.26.0"
   ```

2. **Update LibRaw Submodule** (if needed)
   
   ```bash
   cd external/LibRaw
   git fetch --tags
   git checkout 0.21.2  # or desired stable tag
   cd ../..
   git add external/LibRaw
   ```

3. **Commit Changes**
   
   ```bash
   git add rawpy/_version.py
   git commit -m "Release vX.Y.Z"
   ```

4. **Create and Push Tag**
   
   ```bash
   git tag vX.Y.Z  # e.g., v0.26.0
   git push origin main --tags
   ```

5. **Automated Build and Release**
   
   GitHub Actions will automatically:
   - Build wheels for all supported platforms (Linux, macOS, Windows)
   - Run tests on all platforms
   - Build documentation
   - Publish wheels to PyPI (if tests pass)
   - Deploy documentation to GitHub Pages

### Creating a Pre-release

Pre-releases follow the same process but use a pre-release version number:

1. **Update Version Number**
   
   Edit `rawpy/_version.py`:
   ```python
   __version__ = "X.Y.Za1"  # Alpha pre-release (e.g., "0.26.0a1")
   # or
   __version__ = "X.Y.Zb1"  # Beta pre-release (e.g., "0.26.0b1")
   ```

2. **Update LibRaw to Snapshot** (if desired)
   
   ```bash
   cd external/LibRaw
   git fetch
   git checkout main  # or a specific commit
   cd ../..
   git add external/LibRaw
   ```

3. **Follow steps 3-5 from stable release process**

### Installing Pre-releases

Users can install pre-releases with:
```bash
pip install --pre rawpy
```

### CI/CD Workflow

The `.github/workflows/ci.yml` workflow handles:

1. **Build Job**: Builds wheels for:
   - Linux (x86_64 and aarch64)
   - macOS (Apple Silicon)
   - Windows (x86_64)
   - Python versions: 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14 (when available)

2. **Test Job**: Tests all built wheels across platforms

3. **Docs Job**: Builds Sphinx documentation

4. **Publish Jobs**: 
   - Only triggered on tag push (`refs/tags/v*`)
   - Publishes wheels to PyPI
   - Deploys documentation to GitHub Pages

## Environment Variables

### Build-time Variables

- `RAWPY_BUILD_GPL_CODE=1` - Include GPL demosaic packs (macOS/Windows only)
- `RAWPY_USE_SYSTEM_LIBRAW=1` - Use system LibRaw instead of building from source
- `PKG_CONFIG` - Path to pkg-config binary
- `PYTHON_VERSION` - Python version for build scripts (e.g., "3.12")
- `PYTHON_ARCH` - Architecture (e.g., "x86_64", "aarch64", "arm64")
- `NUMPY_VERSION` - NumPy version constraint (e.g., "2.0.*")
- `MACOS_MIN_VERSION` - Minimum macOS version for build (e.g., "11.0")

## Troubleshooting

### Import Error on Linux

If you get "ImportError: libraw.so: cannot open shared object file":

```bash
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/99local.conf
sudo ldconfig
```

### Build Failures

- Ensure all git submodules are initialized: `git submodule update --init`
- Check that you have the required compilers installed
- On Windows, ensure Visual Studio environment is properly set up

### Test Failures

- Ensure you have test dependencies installed: `pip install -r dev-requirements.txt`
- Check that test data files are present in `test/` directory

## Security

The CI workflow implements several security best practices:

- Read-only GITHUB_TOKEN permissions by default
- Minimal use of third-party actions
- Separate jobs for testing and building to prevent dependency contamination
- Trusted publishing to PyPI using OIDC (no API tokens required)

## Getting Help

- **Issues**: https://github.com/letmaik/rawpy/issues
- **API Documentation**: https://letmaik.github.io/rawpy/api/
- **Tutorials**: https://github.com/letmaik/rawpy-notebooks

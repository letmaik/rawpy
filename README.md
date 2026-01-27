<p align="center">
<a href="https://en.wikipedia.org/wiki/Bayer_filter"><img width="500" src="https://raw.githubusercontent.com/letmaik/rawpy/main/logo/logo.png"></a>
</p>

rawpy is an easy-to-use Python wrapper for the [LibRaw library][libraw].
It also contains some extra functionality for finding and repairing hot/dead pixels.

[API Documentation](https://letmaik.github.io/rawpy/api/)

[Jupyter notebook tutorials](https://github.com/letmaik/rawpy-notebooks/blob/master/README.md)

## Sample code

Load a RAW file and save the postprocessed image using default [parameters](https://letmaik.github.io/rawpy/api/rawpy.Params.html):

```python
import rawpy
import imageio.v3 as iio

path = 'image.nef'
with rawpy.imread(path) as raw:
    rgb = raw.postprocess()
iio.imwrite('default.tiff', rgb)
```

Save as 16-bit linear image:

```python
with rawpy.imread(path) as raw:
    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
iio.imwrite('linear.tiff', rgb)
```

Extract embedded thumbnail/preview image and save as JPEG:

```python
with rawpy.imread(path) as raw:
    # raises rawpy.LibRawNoThumbnailError if thumbnail missing
    # raises rawpy.LibRawUnsupportedThumbnailError if unsupported format
    thumb = raw.extract_thumb()
if thumb.format == rawpy.ThumbFormat.JPEG:
    # thumb.data is already in JPEG format, save as-is
    with open('thumb.jpeg', 'wb') as f:
        f.write(thumb.data)
elif thumb.format == rawpy.ThumbFormat.BITMAP:
    # thumb.data is an RGB numpy array, convert with imageio
    iio.imwrite('thumb.jpeg', thumb.data)
```

Find bad pixels using multiple RAW files and repair them:

```python
import rawpy.enhance

paths = ['image1.nef', 'image2.nef', 'image3.nef']
bad_pixels = rawpy.enhance.find_bad_pixels(paths)

for path in paths:
    with rawpy.imread(path) as raw:
        rawpy.enhance.repair_bad_pixels(raw, bad_pixels, method='median')
        rgb = raw.postprocess()
    iio.imwrite(path + '.tiff', rgb)
```

## Installation

Install rawpy by running:
```sh
pip install rawpy
```

64-bit binary wheels are provided for Linux, macOS, and Windows.

### Stable vs. pre-release

All stable rawpy releases are always built against a stable LibRaw library release.
You can output the LibRaw version with `print(rawpy.libraw_version)`.

rawpy pre-releases have version numbers like `0.15.0a1` and are built against
a recent LibRaw snapshot. To install a pre-release, run:
```sh
pip install --pre rawpy
```

### Optional features

The underlying [LibRaw library][libraw] supports several optional features.
The following table shows which PyPI binary wheels support which features.

| Feature            | Windows | macOS | Linux |
| ------------------ | ------- | ----- | ----- |
| LCMS color engine  | yes     | yes   | yes   |
| RedCine codec      | yes     | yes   | yes   |
| DNG deflate codec  | yes     | yes   | yes   |
| DNG lossy codec    | yes     | yes   | yes   |
| Demosaic Pack GPL2 | no      | no    | no    |
| Demosaic Pack GPL3 | no      | no    | no    |
| OpenMP             | yes     | no    | yes   |

Tip: You can dynamically query supported features by inspecting the `rawpy.flags` dictionary.

Note on GPL demosaic packs: The GPL2 and GPL3 demosaic packs are not included as rawpy is licensed
under the MIT license which is incompatible with GPL.

### Installation from source on Linux/macOS

For macOS, LibRaw is built as part of the rawpy build (see external/).
For Linux, you need to install the LibRaw library on your system.

On Ubuntu, you can get (an outdated) version with:

```sh
sudo apt-get install libraw-dev
```

Or install the latest release version from the source repository:

```sh
git clone https://github.com/LibRaw/LibRaw.git libraw
git clone https://github.com/LibRaw/LibRaw-cmake.git libraw-cmake
cd libraw
git checkout 0.20.0
cp -R ../libraw-cmake/* .
cmake .
sudo make install
```
    
After that, install rawpy using:

```sh
git clone https://github.com/letmaik/rawpy
cd rawpy
pip install numpy cython
pip install .
```
    
On Linux, if you get the error "ImportError: libraw.so: cannot open shared object file: No such file or directory"
when trying to use rawpy, then do the following:

```sh
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/99local.conf
sudo ldconfig
```

The LibRaw library is installed in /usr/local/lib (if installed manually) and apparently this folder is not searched
for libraries by default in some Linux distributions.

### Building wheels locally

The CI build scripts can be used to build wheels locally on Linux, macOS, and Windows.
These scripts are designed to work in both CI and local development environments.

#### macOS

Requirements:
- Xcode command line tools (`xcode-select --install`)
- CMake (`brew install cmake` or download from https://cmake.org/)

Set the required environment variables and run the build script:

```sh
export MACOS_MIN_VERSION=11.0
export PYTHON_ARCH=arm64  # or x86_64 for Intel Macs
export PYTHON_VERSION=3.9  # or any supported version (3.9-3.14)
export NUMPY_VERSION=2.0.*
.github/scripts/build-macos.sh
```

The script will:
- Clone the multibuild repository if not already present
- Download and compile dependencies (libjpeg-turbo, libjasper, lcms2) if not already installed
- Build the rawpy wheel in the `dist/` directory

On subsequent runs, the script will skip downloading and building dependencies that are already present,
making rebuilds much faster.

#### Linux

For Linux, the build scripts use a manylinux Docker container to ensure broad compatibility:

```sh
docker run --rm \
  -e PYTHON_VERSION=3.9 \
  -e NUMPY_VERSION=2.0.* \
  -e PYTHON_ARCH=x86_64 \
  -v `pwd`:/io \
  quay.io/pypa/manylinux_2_28_x86_64 \
  /io/.github/scripts/build-linux.sh
```

The script will:
- Install required system dependencies
- Download and compile dependencies (libjpeg-turbo, libjasper, lcms2) if not already installed
- Build and repair the rawpy wheel using auditwheel
- Place the wheel in the `dist/` directory

Like the macOS script, subsequent runs will skip dependencies that are already installed.

#### Windows

Requirements:
- Visual Studio (with C++ build tools)
- PowerShell

In a PowerShell window:
```powershell
$env:USE_CONDA = '1'
$env:PYTHON_VERSION = '3.9'
$env:PYTHON_ARCH = '64'
$env:NUMPY_VERSION = '2.0.*'
.github/scripts/build-windows.ps1
```
The above will download all build dependencies (including a Python installation)
and is fully configured through the four environment variables.
Set `USE_CONDA = '0'` to build within an existing Python environment.

#### Troubleshooting

If you encounter issues during local builds:

1. **"fatal: destination path 'multibuild' already exists"** (macOS): This is now handled automatically.
   The script checks if the directory exists before cloning.

2. **Library already installed errors**: The scripts now check for existing installations and skip
   rebuilding dependencies that are already present.

3. **Clean rebuild**: If you want to force a complete rebuild from scratch, you can remove:
   - macOS: `multibuild/`, `external/libs/`, and any `*.tar.gz` files
   - Linux: Remove the Docker container and start fresh

4. **Build failures**: Check that all required environment variables are set correctly and that
   you have the necessary build tools installed for your platform.

### Installation from source on Windows (alternative method)

These instructions are experimental and support is not provided for them.
Typically, there should be no need to build manually since wheels are hosted on PyPI.

You need to have Visual Studio installed to build rawpy.

In a PowerShell window:
```sh
$env:USE_CONDA = '1'
$env:PYTHON_VERSION = '3.7'
$env:PYTHON_ARCH = '64'
$env:NUMPY_VERSION = '1.14.*'
git clone https://github.com/letmaik/rawpy
cd rawpy
.github/scripts/build-windows.ps1
```
The above will download all build dependencies (including a Python installation)
and is fully configured through the four environment variables.
Set `USE_CONDA = '0'` to build within an existing Python environment.

## FAQ

### I'm getting "LibRawFileUnsupportedError: Unsupported file format or not RAW file"

This error occurs when rawpy/LibRaw cannot recognize the file as a supported RAW image format. Common causes include:

1. **The file is not actually a RAW file** - Make sure you're trying to open a RAW image file (e.g., .NEF, .CR2, .ARW, .DNG, etc.) and not a regular image format like JPEG or PNG.

2. **The file is corrupted or incomplete** - If the file was not fully downloaded or is damaged, LibRaw cannot read it properly.

3. **The file lacks proper headers** - Some proprietary or headerless RAW formats are not supported by LibRaw. RAW files need to contain proper metadata headers that identify the camera model, sensor configuration, and other essential information for LibRaw to decode them.

4. **Unsupported camera or RAW format** - While LibRaw supports a [wide range of cameras](https://www.libraw.org/supported-cameras), some very new or obscure camera models may not be supported yet. Check the LibRaw website for the list of supported cameras.

**What you can do:**
- Verify the file is a genuine RAW file from a supported camera
- Try opening the file with the camera manufacturer's software to confirm it's valid
- Check if you're using the latest version of rawpy, as newer versions may support additional cameras
- If you have a headerless or proprietary RAW format, you may need to convert it to a standard format like DNG using the camera manufacturer's tools first

[libraw]: https://www.libraw.org

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

### Installation from source on Windows

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


[libraw]: https://www.libraw.org

rawpy
=====

.. image:: https://travis-ci.org/letmaik/rawpy.svg?branch=master
    :target: https://travis-ci.org/letmaik/rawpy
    :alt: Linux/macOS Build Status
   
.. image:: https://ci.appveyor.com/api/projects/status/f8ibd8mejxs9xq5w?svg=true
    :target: https://ci.appveyor.com/project/letmaik/rawpy
    :alt: Windows Build Status

rawpy is an easy-to-use Python wrapper for the LibRaw_ library.
It also contains some extra functionality for finding and repairing hot/dead pixels.

`API Documentation <https://letmaik.github.io/rawpy/api/>`_

`Jupyter notebook tutorials <https://github.com/letmaik/rawpy-notebooks/blob/master/README.md>`_

Sample code
-----------

Load a RAW file and save the postprocessed image using default parameters:

.. code-block:: python

	import rawpy
	import imageio
	
	path = 'image.nef'
	with rawpy.imread(path) as raw:
	    rgb = raw.postprocess()
	imageio.imsave('default.tiff', rgb)
	
Save as 16-bit linear image:

.. code-block:: python
	
	with rawpy.imread(path) as raw:
	    rgb = raw.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16)
	imageio.imsave('linear.tiff', rgb)

Find bad pixels using multiple RAW files and repair them:

.. code-block:: python

	import rawpy.enhance
	
	paths = ['image1.nef', 'image2.nef', 'image3.nef']
	bad_pixels = rawpy.enhance.find_bad_pixels(paths)
	
	for path in paths:
	    with rawpy.imread(path) as raw:
	        rawpy.enhance.repair_bad_pixels(raw, bad_pixels, method='median')
	        rgb = raw.postprocess()
	    imageio.imsave(path + '.tiff', rgb)

Installation
------------

Binary wheels for Linux, macOS, and Windows are provided for Python 2.7, 3.4, 3.5, and 3.6.
These can be installed with a simple ``pip install rawpy``.
Currently, Linux and macOS wheels are only available as 64 bit versions.

Installation from source on Linux/macOS
---------------------------------------

If you have the need to use a specific libraw version or you can't use the provided binary wheels
then follow the steps in this section to build rawpy from source.

First, install the LibRaw_ library on your system.

On Ubuntu, you can get (an outdated) version with:

.. code-block:: sh

    sudo apt-get install libraw-dev
    
Or install the latest release version from the source repository:

.. code-block:: sh

    git clone https://github.com/LibRaw/LibRaw.git libraw
    git clone https://github.com/LibRaw/LibRaw-cmake.git libraw-cmake
    cd libraw
    git checkout 0.18.2
    cp -R ../libraw-cmake/* .
    cmake .
    sudo make install
    
After that, install rawpy using ``pip install rawpy --no-binary :all:``.
    
If you get the error "ImportError: libraw.so: cannot open shared object file: No such file or directory"
when trying to use rawpy, then do the following:

.. code-block:: sh

    echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/99local.conf
    sudo ldconfig

The LibRaw library is installed in /usr/local/lib (if installed manually) and apparently this folder is not searched
for libraries by default in some Linux distributions.

NumPy Dependency
----------------

rawpy depends on NumPy. The minimum supported NumPy version depends on your Python version:

========== =========
Python     NumPy
---------- ---------
2.7        >= 1.7
3.4        >= 1.8
3.5        >= 1.9
3.6        >= 1.11
========== =========

.. _LibRaw: http://www.libraw.org

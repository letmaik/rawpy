rawpy
=====

.. image:: https://travis-ci.org/neothemachine/rawpy.svg?branch=master
    :target: https://travis-ci.org/neothemachine/rawpy
    :alt: Linux Build Status

.. image:: https://travis-ci.org/neothemachine/rawpy.svg?branch=mac-wheels
    :target: https://travis-ci.org/neothemachine/rawpy
    :alt: Mac OS X Build Status
    
.. image:: https://ci.appveyor.com/api/projects/status/f8ibd8mejxs9xq5w/branch/master
    :target: https://ci.appveyor.com/project/neothemachine/rawpy/branch/master
    :alt: Windows Build Status

rawpy is an easy-to-use Python wrapper for the LibRaw_ library.
It also contains some extra functionality for finding and repairing hot/dead pixels.

`API Documentation <http://pythonhosted.org/rawpy/api/>`_

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

NumPy Dependency
----------------

Before installing rawpy, you need to have *numpy* installed.
You can check your numpy version with ``pip freeze``.

The minimum supported numpy version depends on your Python version:

========== =========
Python     numpy
---------- ---------
2.7 - 3.3  >= 1.7.1
3.4        >= 1.8.1
3.5        >= 1.9.3
========== =========

You can install numpy with ``pip install numpy``.

Installation on Windows and Mac OS X
------------------------------------

Binaries are provided for Python 2.7, 3.3, 3.4 and 3.5 for both 32 and 64 bit.
These can be installed with a simple ``pip install --use-wheel rawpy`` 
(or just ``pip install rawpy`` if using pip >= 1.5).

Installation on Linux
---------------------

You need to have the LibRaw_ library installed to use this wrapper.

On Ubuntu, you can get (an outdated) version with:

.. code-block:: sh

    sudo apt-get install libraw-dev
    
Or install the latest release version from the source repository:

.. code-block:: sh

    git clone git://github.com/LibRaw/LibRaw.git libraw
    git clone git://github.com/LibRaw/LibRaw-cmake.git libraw-cmake
    cd libraw
    git checkout 0.17.1
    cp -R ../libraw-cmake/* .
    cmake .
    sudo make install
    
After that, it's the usual ``pip install rawpy``.
    
If you get the error "ImportError: libraw.so: cannot open shared object file: No such file or directory"
when trying to use rawpy, then do the following:

.. code-block:: sh

    echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/99local.conf
    sudo ldconfig

The LibRaw library is installed in /usr/local/lib and apparently this folder is not searched
for libraries by default in some Linux distributions.

.. _LibRaw: http://www.libraw.org

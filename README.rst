rawpy
=====

.. image:: https://travis-ci.org/neothemachine/rawpy.svg?branch=master
    :target: https://travis-ci.org/neothemachine/rawpy
    :alt: Build Status
    
.. image:: https://ci.appveyor.com/api/projects/status/f8ibd8mejxs9xq5w/branch/master
    :target: https://ci.appveyor.com/project/neothemachine/rawpy/branch/master
    :alt: Build Status

rawpy is an easy-to-use Python wrapper for the LibRaw_ library.

NOTE: This is a work-in-progress and not usable yet!

Sample code
-----------

TODO

Installation on Linux
---------------------

You need to have the LibRaw_ library installed to use this wrapper.

On Ubuntu, you can get (an outdated) version with:

.. code-block:: sh

    sudo apt-get install libraw-dev
    
Or install the latest developer version from the SVN repository:

.. code-block:: sh

    git clone git://github.com/LibRaw/LibRaw.git libraw
    cd libraw
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

Installation on Windows
-----------------------

Binaries are provided for Python 2.7 and 3.4 for both 32 and 64 bit.
These can be installed with a simple ``pip install --use-wheel rawpy`` 
(or just ``pip install rawpy`` if using pip >= 1.5).

Binaries for other Python versions are currently not produced but if there is a need
then this is possible as well. In that case, just contact me.

.. _LibRaw: http://www.libraw.org

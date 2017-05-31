from __future__ import absolute_import

from ._version import __version__

import rawpy._rawpy
globals().update({k:v for k,v in rawpy._rawpy.__dict__.items() if not k.startswith('_')})

def imread(pathOrFile):
    """
    Convenience function that creates a :class:`rawpy.RawPy` instance, opens the given file,
    unpacks the RAW data, and returns the :class:`rawpy.RawPy` instance for further processing.
    
    :param str|file pathOrFile: path or file object of RAW image that will be read
    :rtype: :class:`rawpy.RawPy`
    """
    d = RawPy()
    if hasattr(pathOrFile, 'read'):
        d.open_buffer(pathOrFile)
    else:
        d.open_file(pathOrFile)
    d.unpack()
    return d
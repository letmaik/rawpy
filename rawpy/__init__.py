from __future__ import absolute_import

from ._version import __version__, __version_info__

import rawpy._rawpy
globals().update({k:v for k,v in rawpy._rawpy.__dict__.items() if not k.startswith('_')})

def imread(path):
    """
    Convenience function that creates a :class:`rawpy.RawPy` instance, opens the given file,
    unpacks the RAW data, and returns the :class:`rawpy.RawPy` instance for further processing.
    
    :param str path: path to RAW image that will be opened
    :rtype: :class:`rawpy.RawPy`
    """
    d = RawPy()
    d.open_file(path)
    d.unpack()
    return d
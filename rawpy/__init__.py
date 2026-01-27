from __future__ import absolute_import

from ._version import __version__

import rawpy._rawpy
globals().update({k:v for k,v in rawpy._rawpy.__dict__.items() if not k.startswith('_')})

def imread(pathOrFile, shot_select=0):
    """
    Convenience function that creates a :class:`rawpy.RawPy` instance, opens the given file,
    and returns the :class:`rawpy.RawPy` instance for further processing.
    
    :param str|file pathOrFile: path or file object of RAW image that will be read
    :param int shot_select: select which image to extract from RAW files that contain multiple images
                            (e.g., Dual Pixel RAW). Default is 0 for the first/main image.
    :rtype: :class:`rawpy.RawPy`
    """
    d = RawPy()
    if hasattr(pathOrFile, 'read'):
        d.open_buffer(pathOrFile, shot_select=shot_select)
    else:
        d.open_file(pathOrFile, shot_select=shot_select)
    return d
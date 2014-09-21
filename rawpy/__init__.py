from __future__ import absolute_import

import rawpy._rawpy
globals().update(rawpy._rawpy.__dict__)

def imread(path):
    """
    Convenience function that creates a RawPy instance, opens the given file,
    unpacks the RAW data, and returns the RawPy instance for further processing.
    """
    d = RawPy()
    d.open_file(path)
    d.unpack()
    return d
from __future__ import absolute_import, annotations

from typing import TYPE_CHECKING, Union, BinaryIO

from ._version import __version__

import rawpy._rawpy
globals().update({k:v for k,v in rawpy._rawpy.__dict__.items() if not k.startswith('_')})

if TYPE_CHECKING:
    # Import types for type checking only (no runtime overhead)
    from rawpy._rawpy import (
        RawPy,
        RawType,
        ThumbFormat,
        DemosaicAlgorithm,
        FBDDNoiseReductionMode,
        ColorSpace,
        HighlightMode,
        Params,
        ImageSizes,
        Thumbnail,
        LibRawError,
        LibRawFatalError,
        LibRawNonFatalError,
        LibRawUnspecifiedError,
        LibRawFileUnsupportedError,
        LibRawRequestForNonexistentImageError,
        LibRawOutOfOrderCallError,
        LibRawNoThumbnailError,
        LibRawUnsupportedThumbnailError,
        LibRawInputClosedError,
        LibRawNotImplementedError,
        LibRawUnsufficientMemoryError,
        LibRawDataError,
        LibRawIOError,
        LibRawCancelledByCallbackError,
        LibRawBadCropError,
        LibRawTooBigError,
        LibRawMemPoolOverflowError,
        NotSupportedError,
    )

def imread(pathOrFile: Union[str, BinaryIO], shot_select: int = 0) -> RawPy:
    """
    Convenience function that creates a :class:`rawpy.RawPy` instance, opens the given file,
    and returns the :class:`rawpy.RawPy` instance for further processing.
    
    :param pathOrFile: path or file object of RAW image that will be read
    :type pathOrFile: str or file-like object
    :param shot_select: select which image to extract from RAW files that contain multiple images
                        (e.g., Dual Pixel RAW). Default is 0 for the first/main image.
    :type shot_select: int
    :return: RawPy instance with the opened RAW image
    :rtype: rawpy.RawPy
    """
    d = RawPy()
    if hasattr(pathOrFile, 'read'):
        d.open_buffer(pathOrFile)
    else:
        d.open_file(pathOrFile)
    d.set_unpack_params(shot_select=shot_select)
    return d
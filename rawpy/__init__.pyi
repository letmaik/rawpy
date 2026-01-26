"""Type stubs for rawpy"""
from __future__ import annotations

from enum import Enum
from typing import Any, BinaryIO, NamedTuple, Optional, Sequence, Tuple, Union, overload
import numpy as np
from numpy.typing import NDArray

# Module-level version
__version__: str
libraw_version: Tuple[int, int, int]
flags: Optional[dict[str, bool]]

# Named tuples
class ImageSizes(NamedTuple):
    raw_height: int
    raw_width: int
    height: int
    width: int
    top_margin: int
    left_margin: int
    iheight: int
    iwidth: int
    pixel_aspect: float
    flip: int
    crop_left_margin: int
    crop_top_margin: int
    crop_width: int
    crop_height: int

class Thumbnail(NamedTuple):
    format: ThumbFormat
    data: Union[bytes, NDArray[np.uint8]]

# Enums
class RawType(Enum):
    """RAW image type."""
    Flat: int
    """Bayer type or black and white"""
    Stack: int
    """Foveon and other multi-color images"""

class ThumbFormat(Enum):
    """Thumbnail format."""
    JPEG: int
    """JPEG image as bytes object."""
    BITMAP: int
    """RGB image as ndarray object."""

class DemosaicAlgorithm(Enum):
    """Demosaic algorithms."""
    LINEAR: int
    VNG: int
    PPG: int
    AHD: int
    DCB: int
    MODIFIED_AHD: int
    AFD: int
    VCD: int
    MIXED_VCD_MODIFIED_AHD: int
    LMMSE: int
    AMAZE: int
    DHT: int
    AAHD: int
    
    @property
    def isSupported(self) -> Optional[bool]: ...
    def checkSupported(self) -> None: ...

class FBDDNoiseReductionMode(Enum):
    """FBDD noise reduction mode."""
    Off: int
    Light: int
    Full: int

class ColorSpace(Enum):
    """Output color space."""
    raw: int
    sRGB: int
    Adobe: int
    Wide: int
    ProPhoto: int
    XYZ: int
    ACES: int

class HighlightMode(Enum):
    """Highlight mode."""
    Clip: int
    Ignore: int
    Blend: int
    
    @staticmethod
    def Reconstruct(level: int) -> int: ...

# Exceptions
class LibRawError(Exception):
    """Base class for LibRaw errors."""
    ...

class LibRawFatalError(LibRawError):
    """Fatal LibRaw error."""
    ...

class LibRawNonFatalError(LibRawError):
    """Non-fatal LibRaw error."""
    ...

class LibRawUnspecifiedError(LibRawNonFatalError):
    ...

class LibRawFileUnsupportedError(LibRawNonFatalError):
    ...

class LibRawRequestForNonexistentImageError(LibRawNonFatalError):
    ...

class LibRawOutOfOrderCallError(LibRawNonFatalError):
    ...

class LibRawNoThumbnailError(LibRawNonFatalError):
    ...

class LibRawUnsupportedThumbnailError(LibRawNonFatalError):
    ...

class LibRawInputClosedError(LibRawNonFatalError):
    ...

class LibRawNotImplementedError(LibRawNonFatalError):
    ...

class LibRawUnsufficientMemoryError(LibRawFatalError):
    ...

class LibRawDataError(LibRawFatalError):
    ...

class LibRawIOError(LibRawFatalError):
    ...

class LibRawCancelledByCallbackError(LibRawFatalError):
    ...

class LibRawBadCropError(LibRawFatalError):
    ...

class LibRawTooBigError(LibRawFatalError):
    ...

class LibRawMemPoolOverflowError(LibRawFatalError):
    ...

class NotSupportedError(Exception):
    """Feature not supported in the current LibRaw version."""
    ...

# Params class
class Params:
    """A class that handles postprocessing parameters."""
    
    def __init__(
        self,
        demosaic_algorithm: Optional[DemosaicAlgorithm] = None,
        half_size: bool = False,
        four_color_rgb: bool = False,
        dcb_iterations: int = 0,
        dcb_enhance: bool = False,
        fbdd_noise_reduction: FBDDNoiseReductionMode = ...,
        noise_thr: Optional[float] = None,
        median_filter_passes: int = 0,
        use_camera_wb: bool = False,
        use_auto_wb: bool = False,
        user_wb: Optional[Sequence[float]] = None,
        output_color: ColorSpace = ...,
        output_bps: int = 8,
        user_flip: Optional[int] = None,
        user_black: Optional[int] = None,
        user_sat: Optional[int] = None,
        no_auto_bright: bool = False,
        auto_bright_thr: Optional[float] = None,
        adjust_maximum_thr: float = 0.75,
        bright: float = 1.0,
        highlight_mode: Union[HighlightMode, int] = ...,
        exp_shift: Optional[float] = None,
        exp_preserve_highlights: float = 0.0,
        no_auto_scale: bool = False,
        gamma: Optional[Tuple[float, float]] = None,
        chromatic_aberration: Optional[Tuple[float, float]] = None,
        bad_pixels_path: Optional[str] = None,
    ) -> None: ...

# Main RawPy class
class RawPy:
    """
    Load RAW images, work on their data, and create a postprocessed (demosaiced) image.
    
    All operations are implemented using numpy arrays.
    """
    
    def __enter__(self) -> RawPy: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    
    def close(self) -> None:
        """Release all resources and close the RAW image."""
        ...
    
    def open_file(self, path: str) -> None:
        """
        Opens the given RAW image file. Should be followed by a call to unpack().
        
        .. NOTE:: This is a low-level method, consider using imread() instead.
        
        :param path: The path to the RAW image.
        """
        ...
    
    def open_buffer(self, fileobj: BinaryIO) -> None:
        """
        Opens the given RAW image file-like object. Should be followed by a call to unpack().
        
        .. NOTE:: This is a low-level method, consider using imread() instead.
        
        :param fileobj: The file-like object.
        """
        ...
    
    def unpack(self) -> None:
        """
        Unpacks/decodes the opened RAW image.
        
        .. NOTE:: This is a low-level method, consider using imread() instead.
        """
        ...
    
    def unpack_thumb(self) -> None:
        """
        Unpacks/decodes the thumbnail/preview image, whichever is bigger.
        
        .. NOTE:: This is a low-level method, consider using extract_thumb() instead.
        """
        ...
    
    @property
    def raw_type(self) -> RawType:
        """
        Return the RAW type.
        
        :rtype: RawType
        """
        ...
    
    @property
    def raw_image(self) -> NDArray[np.uint16]:
        """
        View of RAW image. Includes margin.

        For Bayer images, a 2D ndarray is returned.
        For Foveon and other RGB-type images, a 3D ndarray is returned.
        Note that there may be 4 color channels, where the 4th channel can be blank (zeros).
        
        Modifying the returned array directly influences the result of calling postprocess().
        
        .. WARNING:: The returned numpy array can only be accessed while this RawPy instance
            is not closed yet, that is, within a with block or before calling close().
            If you need to work on the array after closing the RawPy instance,
            make sure to create a copy of it with raw_image = raw.raw_image.copy().
        
        :rtype: ndarray of shape (h,w[,c])
        """
        ...
    
    @property
    def raw_image_visible(self) -> NDArray[np.uint16]:
        """
        Like raw_image but without margin.
        
        :rtype: ndarray of shape (hv,wv[,c])
        """
        ...
    
    def raw_value(self, row: int, column: int) -> int:
        """
        Return RAW value at given position relative to the full RAW image.
        Only usable for flat RAW images (see raw_type property).
        """
        ...
    
    def raw_value_visible(self, row: int, column: int) -> int:
        """
        Return RAW value at given position relative to visible area of image.
        Only usable for flat RAW images (see raw_type property).
        """
        ...
    
    @property
    def sizes(self) -> ImageSizes:
        """
        Return an ImageSizes instance with size information of
        the RAW image and postprocessed image.
        """
        ...
    
    @property
    def num_colors(self) -> int:
        """
        Number of colors.
        Note that e.g. for RGBG this can be 3 or 4, depending on the camera model,
        as some use two different greens.
        """
        ...
    
    @property
    def color_desc(self) -> bytes:
        """
        String description of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG).
        Note that same letters may not refer strictly to the same color.
        There are cameras with two different greens for example.
        """
        ...
    
    def raw_color(self, row: int, column: int) -> int:
        """
        Return color index for the given coordinates relative to the full RAW size.
        Only usable for flat RAW images (see raw_type property).
        """
        ...
    
    @property
    def raw_colors(self) -> NDArray[np.uint8]:
        """
        An array of color indices for each pixel in the RAW image.
        Equivalent to calling raw_color(y,x) for each pixel.
        Only usable for flat RAW images (see raw_type property).
        
        :rtype: ndarray of shape (h,w)
        """
        ...
    
    @property
    def raw_colors_visible(self) -> NDArray[np.uint8]:
        """
        Like raw_colors but without margin.
        
        :rtype: ndarray of shape (hv,wv)
        """
        ...
    
    @property
    def raw_pattern(self) -> Optional[NDArray[np.uint8]]:
        """
        The smallest possible Bayer pattern of this image.
        
        :rtype: ndarray, or None if not a flat RAW image
        """
        ...
    
    @property
    def camera_whitebalance(self) -> list[float]:
        """
        White balance coefficients (as shot). Either read from file or calculated.
        
        :rtype: list of length 4
        """
        ...
    
    @property
    def daylight_whitebalance(self) -> list[float]:
        """
        White balance coefficients for daylight (daylight balance). 
        Either read from file, or calculated on the basis of file data, 
        or taken from hardcoded constants.
        
        :rtype: list of length 4
        """
        ...
    
    @property
    def black_level_per_channel(self) -> list[int]:
        """
        Per-channel black level correction.
        
        :rtype: list of length 4
        """
        ...
    
    @property
    def white_level(self) -> int:
        """
        Level at which the raw pixel value is considered to be saturated.
        """
        ...
    
    @property
    def camera_white_level_per_channel(self) -> Optional[list[int]]:
        """
        Per-channel saturation levels read from raw file metadata, if it exists. Otherwise None.

        :rtype: list of length 4, or None if metadata missing
        """
        ...
    
    @property
    def color_matrix(self) -> NDArray[np.float32]:
        """
        Color matrix, read from file for some cameras, calculated for others. 
        
        :rtype: ndarray of shape (3,4)
        """
        ...
    
    @property
    def rgb_xyz_matrix(self) -> NDArray[np.float32]:
        """
        Camera RGB - XYZ conversion matrix.
        This matrix is constant (different for different models).
        Last row is zero for RGB cameras and non-zero for different color models (CMYG and so on).
        
        :rtype: ndarray of shape (4,3)
        """
        ...
    
    @property
    def tone_curve(self) -> NDArray[np.uint16]:
        """
        Camera tone curve, read from file for Nikon, Sony and some other cameras.
        
        :rtype: ndarray of length 65536
        """
        ...
    
    def dcraw_process(self, params: Optional[Params] = None, **kw: Any) -> None:
        """
        Postprocess the currently loaded RAW image.
        
        .. NOTE:: This is a low-level method, consider using postprocess() instead.
        
        :param params: The parameters to use for postprocessing.
        :param kw: Alternative way to provide postprocessing parameters.
        """
        ...
    
    @overload
    def postprocess(self, params: Params) -> NDArray[np.uint8]: ...
    @overload
    def postprocess(
        self,
        *,
        demosaic_algorithm: Optional[DemosaicAlgorithm] = None,
        half_size: bool = False,
        four_color_rgb: bool = False,
        dcb_iterations: int = 0,
        dcb_enhance: bool = False,
        fbdd_noise_reduction: FBDDNoiseReductionMode = ...,
        noise_thr: Optional[float] = None,
        median_filter_passes: int = 0,
        use_camera_wb: bool = False,
        use_auto_wb: bool = False,
        user_wb: Optional[Sequence[float]] = None,
        output_color: ColorSpace = ...,
        output_bps: int = 8,
        user_flip: Optional[int] = None,
        user_black: Optional[int] = None,
        user_sat: Optional[int] = None,
        no_auto_bright: bool = False,
        auto_bright_thr: Optional[float] = None,
        adjust_maximum_thr: float = 0.75,
        bright: float = 1.0,
        highlight_mode: Union[HighlightMode, int] = ...,
        exp_shift: Optional[float] = None,
        exp_preserve_highlights: float = 0.0,
        no_auto_scale: bool = False,
        gamma: Optional[Tuple[float, float]] = None,
        chromatic_aberration: Optional[Tuple[float, float]] = None,
        bad_pixels_path: Optional[str] = None,
    ) -> NDArray[np.uint8]: ...
    
    def extract_thumb(self) -> Thumbnail:
        """
        Extracts and returns the thumbnail/preview image (whichever is bigger)
        of the opened RAW image as Thumbnail object.
        For JPEG thumbnails, data is a bytes object and can be written as-is to file.
        For bitmap thumbnails, data is an ndarray of shape (h,w,c).
        If no image exists or the format is unsupported, an exception is raised.
        
        :rtype: Thumbnail
        """
        ...

# Module-level functions
@overload
def imread(pathOrFile: str) -> RawPy: ...
@overload
def imread(pathOrFile: BinaryIO) -> RawPy: ...

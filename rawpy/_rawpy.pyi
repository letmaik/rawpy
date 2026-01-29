"""Type stubs for rawpy._rawpy Cython module"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple, List, Union, NamedTuple
import numpy as np
from numpy.typing import NDArray

# Module-level version
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
    """Foveon type or sRAW/mRAW files or RawSpeed decoding"""

class ThumbFormat(Enum):
    """Thumbnail/preview image type."""
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
class LibRawError(Exception): ...
class LibRawFatalError(LibRawError): ...
class LibRawNonFatalError(LibRawError): ...
class LibRawUnspecifiedError(LibRawNonFatalError): ...
class LibRawFileUnsupportedError(LibRawNonFatalError): ...
class LibRawRequestForNonexistentImageError(LibRawNonFatalError): ...
class LibRawOutOfOrderCallError(LibRawNonFatalError): ...
class LibRawNoThumbnailError(LibRawNonFatalError): ...
class LibRawUnsupportedThumbnailError(LibRawNonFatalError): ...
class LibRawInputClosedError(LibRawNonFatalError): ...
class LibRawNotImplementedError(LibRawNonFatalError): ...
class LibRawUnsufficientMemoryError(LibRawFatalError): ...
class LibRawDataError(LibRawFatalError): ...
class LibRawIOError(LibRawFatalError): ...
class LibRawCancelledByCallbackError(LibRawFatalError): ...
class LibRawBadCropError(LibRawFatalError): ...
class LibRawTooBigError(LibRawFatalError): ...
class LibRawMemPoolOverflowError(LibRawFatalError): ...
class NotSupportedError(Exception): ...

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
        fbdd_noise_reduction: FBDDNoiseReductionMode = FBDDNoiseReductionMode.Off,
        noise_thr: Optional[float] = None,
        median_filter_passes: int = 0,
        use_camera_wb: bool = False,
        use_auto_wb: bool = False,
        user_wb: Optional[List[float]] = None,
        output_color: ColorSpace = ColorSpace.sRGB,
        output_bps: int = 8,
        user_flip: Optional[int] = None,
        user_black: Optional[int] = None,
        user_cblack: Optional[List[int]] = None,
        user_sat: Optional[int] = None,
        no_auto_bright: bool = False,
        auto_bright_thr: Optional[float] = None,
        adjust_maximum_thr: float = 0.75,
        bright: float = 1.0,
        highlight_mode: Union[HighlightMode, int] = HighlightMode.Clip,
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
        """
        ...
    
    def open_buffer(self, fileobj: Any) -> None:
        """
        Opens the given RAW image file-like object. Should be followed by a call to unpack().
        """
        ...
    
    def set_unpack_params(self, shot_select: int = 0) -> None:
        """
        Set parameters that affect RAW image unpacking.
        """
        ...
    
    def unpack(self) -> None:
        """
        Unpacks/decodes the opened RAW image.
        """
        ...
    
    def unpack_thumb(self) -> None:
        """
        Unpacks/decodes the thumbnail/preview image, whichever is bigger.
        """
        ...
    
    @property
    def raw_type(self) -> RawType:
        """Return the RAW type."""
        ...
    
    @property
    def raw_image(self) -> NDArray[np.uint16]:
        """
        View of RAW image. Includes margin.
        """
        ...
    
    @property
    def raw_image_visible(self) -> NDArray[np.uint16]:
        """Like raw_image but without margin."""
        ...
    
    def raw_value(self, row: int, column: int) -> int:
        """Return RAW value at given position relative to the full RAW image."""
        ...
    
    def raw_value_visible(self, row: int, column: int) -> int:
        """Return RAW value at given position relative to visible area of image."""
        ...
    
    @property
    def sizes(self) -> ImageSizes:
        """Return size information of the RAW image and postprocessed image."""
        ...
    
    @property
    def num_colors(self) -> int:
        """Number of colors."""
        ...
    
    @property
    def color_desc(self) -> bytes:
        """String description of colors numbered from 0 to 3."""
        ...
    
    def raw_color(self, row: int, column: int) -> int:
        """Return color index for the given coordinates relative to the full RAW size."""
        ...
    
    @property
    def raw_colors(self) -> NDArray[np.uint8]:
        """An array of color indices for each pixel in the RAW image."""
        ...
    
    @property
    def raw_colors_visible(self) -> NDArray[np.uint8]:
        """Like raw_colors but without margin."""
        ...
    
    @property
    def raw_pattern(self) -> Optional[NDArray[np.uint8]]:
        """The smallest possible Bayer pattern of this image."""
        ...
    
    @property
    def camera_whitebalance(self) -> List[float]:
        """White balance coefficients (as shot)."""
        ...
    
    @property
    def daylight_whitebalance(self) -> List[float]:
        """White balance coefficients for daylight."""
        ...
    
    @property
    def auto_whitebalance(self) -> Optional[List[float]]:
        """White balance coefficients used during postprocessing."""
        ...
    
    @property
    def black_level_per_channel(self) -> List[int]:
        """Per-channel black level correction."""
        ...
    
    @property
    def white_level(self) -> int:
        """Level at which the raw pixel value is considered to be saturated."""
        ...
    
    @property
    def camera_white_level_per_channel(self) -> Optional[List[int]]:
        """Per-channel saturation levels read from raw file metadata."""
        ...
    
    @property
    def color_matrix(self) -> NDArray[np.float32]:
        """Color matrix, read from file for some cameras, calculated for others."""
        ...
    
    @property
    def rgb_xyz_matrix(self) -> NDArray[np.float32]:
        """Camera RGB - XYZ conversion matrix."""
        ...
    
    @property
    def tone_curve(self) -> NDArray[np.uint16]:
        """Camera tone curve, read from file for Nikon, Sony and some other cameras."""
        ...
    
    def dcraw_process(self, params: Optional[Params] = None, **kw: Any) -> None:
        """Postprocess the currently loaded RAW image."""
        ...
    
    def postprocess(self, params: Optional[Params] = None, **kw: Any) -> NDArray[np.uint8]:
        """
        Postprocess the currently loaded RAW image and return the
        new resulting image as numpy array.
        """
        ...
    
    def extract_thumb(self) -> Thumbnail:
        """
        Extracts and returns the thumbnail/preview image (whichever is bigger)
        of the opened RAW image as Thumbnail object.
        """
        ...

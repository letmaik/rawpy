"""Type stubs for rawpy._rawpy Cython module"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple, List, Union, NamedTuple, BinaryIO
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
    """
    RAW image type.
    """
    Flat = 0
    """ Bayer type or black and white """
    Stack = 1
    """ Foveon type or sRAW/mRAW files or RawSpeed decoding """

class ThumbFormat(Enum):
    """
    Thumbnail/preview image type.
    """
    JPEG = 1
    """ JPEG image as bytes object. """
    BITMAP = 2
    """ RGB image as ndarray object. """

class DemosaicAlgorithm(Enum):
    """
    Identifiers for demosaic algorithms.
    """
    LINEAR = 0
    VNG = 1
    PPG = 2
    AHD = 3
    DCB = 4
    # 5-9 only usable if demosaic pack GPL2 available
    MODIFIED_AHD = 5
    AFD = 6
    VCD = 7
    VCD_MODIFIED_AHD = 8
    LMMSE = 9
    # 10 only usable if demosaic pack GPL3 available
    AMAZE = 10
    # 11-12 only usable for LibRaw >= 0.16
    DHT = 11
    AAHD = 12
    
    @property
    def isSupported(self) -> Optional[bool]:
        """
        Return True if the demosaic algorithm is supported, False if it is not,
        and None if the support status is unknown. The latter is returned if
        LibRaw < 0.15.4 is used or if it was compiled without cmake.
        
        The necessary information is read from the libraw_config.h header which
        is only written with cmake builds >= 0.15.4.
        """
        ...
    
    def checkSupported(self) -> Optional[bool]:
        """
        Like :attr:`isSupported` but raises an exception for the `False` case.
        """
        ...

class FBDDNoiseReductionMode(Enum):
    """
    FBDD noise reduction modes.
    """
    Off = 0
    Light = 1
    Full = 2

class ColorSpace(Enum):
    """
    Color spaces.
    """
    raw = 0
    sRGB = 1
    Adobe = 2
    Wide = 3
    ProPhoto = 4
    XYZ = 5
    ACES = 6
    P3D65 = 7
    Rec2020 = 8

class HighlightMode(Enum):
    """
    Highlight modes.
    """
    Clip = 0
    Ignore = 1
    Blend = 2
    ReconstructDefault = 5
    
    @classmethod
    def Reconstruct(cls, level: int) -> int:
        """
        :param int level: 3 to 9, low numbers favor whites, high numbers favor colors
        """
        ...

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

class NotSupportedError(Exception):
    def __init__(self, message: str, min_version: Optional[Tuple[int, int, int]] = None) -> None: ...

# Params class
class Params:
    """
    A class that handles postprocessing parameters.
    """
    
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
    ) -> None:
        """
        If use_camera_wb and use_auto_wb are False and user_wb is None, then
        daylight white balance correction is used.
        If both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.
        
        :param rawpy.DemosaicAlgorithm demosaic_algorithm: default is AHD
        :param bool half_size: outputs image in half size by reducing each 2x2 block to one pixel
                               instead of interpolating
        :param bool four_color_rgb: whether to use separate interpolations for two green channels
        :param int dcb_iterations: number of DCB correction passes, requires DCB demosaicing algorithm
        :param bool dcb_enhance: DCB interpolation with enhanced interpolated colors
        :param rawpy.FBDDNoiseReductionMode fbdd_noise_reduction: controls FBDD noise reduction before demosaicing
        :param float noise_thr: threshold for wavelet denoising (default disabled)
        :param int median_filter_passes: number of median filter passes after demosaicing to reduce color artifacts
        :param bool use_camera_wb: whether to use the as-shot white balance values
        :param bool use_auto_wb: whether to try automatically calculating the white balance 
        :param list user_wb: list of length 4 with white balance multipliers for each color 
        :param rawpy.ColorSpace output_color: output color space
        :param int output_bps: 8 or 16
        :param int user_flip: 0=none, 3=180, 5=90CCW, 6=90CW,
                              default is to use image orientation from the RAW image if available
        :param int user_black: custom black level
        :param list user_cblack: list of length 4 with per-channel corrections to user_black.
                                 These are offsets applied on top of user_black for [R, G, B, G2] channels.
        :param int user_sat: saturation adjustment (custom white level)
        :param bool no_auto_scale: Whether to disable pixel value scaling
        :param bool no_auto_bright: whether to disable automatic increase of brightness
        :param float auto_bright_thr: ratio of clipped pixels when automatic brighness increase is used
                                      (see `no_auto_bright`). Default is 0.01 (1%).
        :param float adjust_maximum_thr: see libraw docs
        :param float bright: brightness scaling
        :param highlight_mode: highlight mode
        :type highlight_mode: :class:`rawpy.HighlightMode` | int
        :param float exp_shift: exposure shift in linear scale.
                          Usable range from 0.25 (2-stop darken) to 8.0 (3-stop lighter).
        :param float exp_preserve_highlights: preserve highlights when lightening the image with `exp_shift`.
                          From 0.0 to 1.0 (full preservation).
        :param tuple gamma: pair (power,slope), default is (2.222, 4.5) for rec. BT.709
        :param tuple chromatic_aberration: pair (red_scale, blue_scale), default is (1,1),
                                           corrects chromatic aberration by scaling the red and blue channels
        :param str bad_pixels_path: path to dcraw bad pixels file. Each bad pixel will be corrected using
                                    the mean of the neighbor pixels. See the :mod:`rawpy.enhance` module
                                    for alternative repair algorithms, e.g. using the median.
        """
        ...
    
    # Instance attributes (accessible after __init__)
    user_qual: int
    half_size: bool
    four_color_rgb: bool
    dcb_iterations: int
    dcb_enhance: bool
    fbdd_noise_reduction: int
    noise_thr: float
    median_filter_passes: int
    use_camera_wb: bool
    use_auto_wb: bool
    user_mul: List[float]
    output_color: int
    output_bps: int
    user_flip: int
    user_black: int
    user_cblack: List[int]
    user_sat: int
    no_auto_bright: bool
    auto_bright_thr: float
    adjust_maximum_thr: float
    bright: float
    highlight_mode: int
    exp_shift: float
    exp_preserve_highlights: float
    no_auto_scale: bool
    gamm: Tuple[float, float]
    aber: Tuple[float, float]
    bad_pixels_path: Optional[str]


# Main RawPy class
class RawPy:
    """
    Load RAW images, work on their data, and create a postprocessed (demosaiced) image.
    
    All operations are implemented using numpy arrays.
    """
    
    def __enter__(self) -> RawPy: ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    
    def close(self) -> None:
        """
        Release all resources and close the RAW image.
        
        Consider using context managers for the same effect:
        
        .. code-block:: python
        
            with rawpy.imread('image.nef') as raw:
              # work with raw object
        """
        ...
    
    def open_file(self, path: str) -> None:
        """
        Opens the given RAW image file. Should be followed by a call to :meth:`~rawpy.RawPy.unpack`.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        
        :param str path: The path to the RAW image.
        """
        ...
    
    def open_buffer(self, fileobj: BinaryIO) -> None:
        """
        Opens the given RAW image file-like object. Should be followed by a call to :meth:`~rawpy.RawPy.unpack`.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        
        :param file fileobj: The file-like object.
        """
        ...
    
    def set_unpack_params(self, shot_select: int = 0) -> None:
        """
        Set parameters that affect RAW image unpacking.
        
        This should be called after opening a file and before unpacking.
        
        .. NOTE:: This is a low-level method. When using :func:`rawpy.imread`,
                  unpack parameters can be provided directly.
        
        :param int shot_select: select which image to extract from RAW files that contain multiple images
                                (e.g., Dual Pixel RAW). Default is 0 for the first/main image.
        """
        ...
    
    def unpack(self) -> None:
        """
        Unpacks/decodes the opened RAW image.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        """
        ...
    
    def unpack_thumb(self) -> None:
        """
        Unpacks/decodes the thumbnail/preview image, whichever is bigger.
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.extract_thumb` instead.
        """
        ...
    
    @property
    def raw_type(self) -> RawType:
        """
        Return the RAW type.
        
        :rtype: :class:`rawpy.RawType`
        """
        ...
    
    @property
    def raw_image(self) -> NDArray[np.uint16]:
        """
        View of RAW image. Includes margin.

        For Bayer images, a 2D ndarray is returned.
        For Foveon and other RGB-type images, a 3D ndarray is returned.
        Note that there may be 4 color channels, where the 4th channel can be blank (zeros).
        
        Modifying the returned array directly influences the result of
        calling :meth:`~rawpy.RawPy.postprocess`.
        
        .. WARNING:: The returned numpy array can only be accessed while this RawPy instance
            is not closed yet, that is, within a :code:`with` block or before calling :meth:`~rawpy.RawPy.close`.
            If you need to work on the array after closing the RawPy instance,
            make sure to create a copy of it with :code:`raw_image = raw.raw_image.copy()`.
        
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
        Only usable for flat RAW images (see :attr:`~rawpy.RawPy.raw_type` property).
        """
        ...
    
    def raw_value_visible(self, row: int, column: int) -> int:
        """
        Return RAW value at given position relative to visible area of image.
        Only usable for flat RAW images (see :attr:`~rawpy.RawPy.raw_type` property).        
        """
        ...
    
    @property
    def sizes(self) -> ImageSizes:
        """
        Return a :class:`rawpy.ImageSizes` instance with size information of
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
        
        :rtype: 0 to 3 (sometimes 4)
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
    def camera_whitebalance(self) -> List[float]:
        """
        White balance coefficients (as shot). Either read from file or calculated.
        
        :rtype: list of length 4
        """
        ...
    
    @property
    def daylight_whitebalance(self) -> List[float]:
        """
        White balance coefficients for daylight (daylight balance). 
        Either read from file, or calculated on the basis of file data, 
        or taken from hardcoded constants.
        
        :rtype: list of length 4
        """
        ...
    
    @property
    def auto_whitebalance(self) -> Optional[List[float]]:
        """
        White balance coefficients used during postprocessing.
        
        This property returns the actual white balance multipliers that were used
        during postprocessing, regardless of the white balance mode:
        whether from camera settings, auto white balance calculation, user-specified
        values, or daylight balance.
        
        This property must be accessed after calling :meth:`~rawpy.RawPy.postprocess`
        or :meth:`~rawpy.RawPy.dcraw_process` to get the coefficients that were
        actually applied. If accessed before postprocessing, it returns None.
        
        This corresponds to LibRaw's ``imgdata.color.pre_mul[]`` array after processing,
        which contains the white balance multipliers applied to the raw sensor data.
        
        :rtype: list of length 4, or None if postprocessing hasn't been called yet
        """
        ...
    
    @property
    def black_level_per_channel(self) -> List[int]:
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
    def camera_white_level_per_channel(self) -> Optional[List[int]]:
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
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.postprocess` instead.
        
        :param rawpy.Params params: 
            The parameters to use for postprocessing.
        :param **kw: 
            Alternative way to provide postprocessing parameters.
            The keywords are used to construct a :class:`rawpy.Params` instance.
            If keywords are given, then `params` must be omitted.
        """
        ...
    
    def dcraw_make_mem_image(self) -> NDArray[np.uint8]:
        """
        Return the postprocessed image (see :meth:`~rawpy.RawPy.dcraw_process`) as numpy array.
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.postprocess` instead.
        
        :rtype: ndarray of shape (h,w,c)
        """
        ...
    
    def dcraw_make_mem_thumb(self) -> Thumbnail:
        """
        Return the thumbnail/preview image (see :meth:`~rawpy.RawPy.unpack_thumb`)
        as :class:`rawpy.Thumbnail` object.
        For JPEG thumbnails, data is a bytes object and can be written as-is to file.
        For bitmap thumbnails, data is an ndarray of shape (h,w,c).
        If no image exists or the format is unsupported, an exception is raised.
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.extract_thumb` instead.
        
        :rtype: :class:`rawpy.Thumbnail`
        """
        ...
    
    def postprocess(self, params: Optional[Params] = None, **kw: Any) -> NDArray[np.uint8]:
        """
        Postprocess the currently loaded RAW image and return the
        new resulting image as numpy array.
                
        :param rawpy.Params params: 
            The parameters to use for postprocessing.
        :param **kw: 
            Alternative way to provide postprocessing parameters.
            The keywords are used to construct a :class:`rawpy.Params` instance.
            If keywords are given, then `params` must be omitted.
        :rtype: ndarray of shape (h,w,c)
        """
        ...
    
    def extract_thumb(self) -> Thumbnail:
        """
        Extracts and returns the thumbnail/preview image (whichever is bigger)
        of the opened RAW image as :class:`rawpy.Thumbnail` object.
        
        :rtype: :class:`rawpy.Thumbnail`
        """
        ...

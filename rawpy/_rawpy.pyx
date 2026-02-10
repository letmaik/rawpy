# distutils: language = c++
# cython: embedsignature=True
# cython: language_level=3

from __future__ import print_function

from typing import Optional, Union, Tuple, List, Any, BinaryIO
from numpy.typing import NDArray

from cpython.ref cimport Py_INCREF
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Free
from libc.stddef cimport wchar_t

import numpy as np
from collections import namedtuple
cimport numpy as np
np.import_array()

import os
from enum import Enum

cdef extern from "limits.h":
    cdef unsigned short USHRT_MAX

cdef extern from "Python.h":
    wchar_t* PyUnicode_AsWideCharString(object, Py_ssize_t *)

cdef extern from "def_helper.h":
    cdef int LIBRAW_XTRANS
    
    cdef int _LIBRAW_HAS_FLAGS
    # the following flags are only usable if _LIBRAW_HAS_FLAGS is 1
    # (this is the case for libraw >= 0.15.4 and only when cmake was used)
    cdef int _LIBRAW_USE_DNGLOSSYCODEC
    cdef int _LIBRAW_USE_DNGDEFLATECODEC
    cdef int _LIBRAW_USE_OPENMP
    cdef int _LIBRAW_USE_LCMS
    cdef int _LIBRAW_USE_REDCINECODEC
    cdef int _LIBRAW_USE_RAWSPEED
    cdef int _LIBRAW_USE_DEMOSAIC_PACK_GPL2
    cdef int _LIBRAW_USE_DEMOSAIC_PACK_GPL3
    cdef int _LIBRAW_USE_X3FTOOLS
    cdef int _LIBRAW_USE_6BY9RPI

cdef extern from "data_helper.h":
    ctypedef struct libraw_colordata_black_level_t:
        unsigned cblack[4102]
        unsigned black

    cdef libraw_colordata_black_level_t adjust_bl_(LibRaw* libraw)

cdef extern from "libraw.h":
    ctypedef unsigned short ushort
    
    # some #define's
    cdef int LIBRAW_MAJOR_VERSION
    cdef int LIBRAW_MINOR_VERSION
    cdef int LIBRAW_PATCH_VERSION
    
    cdef float LIBRAW_DEFAULT_AUTO_BRIGHTNESS_THRESHOLD
        
    cdef enum LibRaw_image_formats:
        LIBRAW_IMAGE_JPEG
        LIBRAW_IMAGE_BITMAP
    
    ctypedef struct libraw_raw_inset_crop_t:
        ushort cleft, ctop
        ushort cwidth, cheight
    
    ctypedef struct libraw_image_sizes_t:
        ushort raw_height, raw_width
        ushort height, width
        ushort top_margin, left_margin
        ushort iheight, iwidth
        double pixel_aspect
        int flip
        libraw_raw_inset_crop_t[2] raw_inset_crops
        
    ctypedef struct libraw_colordata_t:
        float       cam_mul[4] 
        float       pre_mul[4]
        ushort      curve[0x10000] # 65536
        unsigned    cblack[4102]
        unsigned    black
        unsigned    maximum
        unsigned    linear_max[4]
        float       cmatrix[3][4]
        float       cam_xyz[4][3]
        void        *profile # a string?
        unsigned    profile_length

    ctypedef struct libraw_rawdata_t:
        ushort *raw_image # 1 component per pixel, for b/w and Bayer type sensors
        ushort (*color4_image)[4] # 4 components per pixel, the 4th component can be void
        ushort (*color3_image)[3] # 3 components per pixel, sRAW/mRAW files, RawSpeed decoding
        libraw_colordata_t          color
        
    ctypedef struct libraw_output_params_t:
        unsigned    greybox[4]     # -A  x1 y1 x2 y2 
        unsigned    cropbox[4]     # -B x1 y1 x2 y2 
        double      aber[4]        # -C 
        double      gamm[6]        # -g 
        float       user_mul[4]    # -r mul0 mul1 mul2 mul3 
        float       bright         # -b 
        float       threshold      #  -n 
        int         half_size      # -h 
        int         four_color_rgb # -f 
        int         highlight      # -H 
        int         use_auto_wb    # -a 
        int         use_camera_wb  # -w 
        int         use_camera_matrix # +M/-M 
        int         output_color   # -o 
        char        *output_profile # -o 
        char        *camera_profile # -p 
        char        *bad_pixels    # -P 
        char        *dark_frame    # -K 
        int         output_bps     # -4 
        int         output_tiff    # -T 
        int         user_flip      # -t 
        int         user_qual      # -q 
        int         user_black     # -k 
        int        user_cblack[4]
        int        sony_arw2_hack
        int         user_sat       # -S 
        int         med_passes     # -m 
        float       auto_bright_thr 
        float       adjust_maximum_thr
        int         no_auto_bright # -W 
        int         use_fuji_rotate# -j 
        int         green_matching
        # DCB parameters 
        int         dcb_iterations
        int         dcb_enhance_fl
        int         fbdd_noiserd
        # VCD parameters 
        int         eeci_refine
        int         es_med_passes
        # AMaZE
        int         ca_correc
        float       cared
        float    cablue
        int cfaline
        float linenoise
        int cfa_clean
        float lclean
        float cclean
        int cfa_green
        float green_thresh
        int exp_correc
        float exp_shift
        float exp_preser
        # WF debanding 
        int   wf_debanding
        float wf_deband_treshold[4]
        # Raw speed 
        int use_rawspeed
        # Disable Auto-scale 
        int no_auto_scale
        # Disable interpolation
        int no_interpolation
        # Disable sRAW YCC to RGB conversion 
        int sraw_ycc
        # Force use x3f data decoding either if demosaic pack GPL2 enabled 
        int force_foveon_x3f

    ctypedef struct libraw_raw_unpack_params_t:
        int use_rawspeed
        int use_dngsdk
        unsigned options
        unsigned shot_select
        unsigned specials
        unsigned max_raw_memory_mb
        int sony_arw2_posterization_thr
        float coolscan_nef_gamma
        char p4shot_order[5]
        char **custom_camera_strings

    ctypedef struct libraw_iparams_t:
        char        make[64]
        char        model[64]
    
        unsigned    raw_count
        unsigned    dng_version
        unsigned    is_foveon
        int         colors
    
        unsigned    filters
        char        xtrans[6][6]
        char        cdesc[5]
        
    ctypedef struct libraw_data_t:
#         ushort                      (*image)[4]
        libraw_image_sizes_t        sizes
        libraw_iparams_t            idata
        libraw_output_params_t        params
        libraw_raw_unpack_params_t  rawparams
#         unsigned int                progress_flags
#         unsigned int                process_warnings
        libraw_colordata_t          color
#         libraw_imgother_t           other
#         libraw_thumbnail_t          thumbnail
        libraw_rawdata_t            rawdata
#         void                *parent_class

    ctypedef struct libraw_processed_image_t:
        LibRaw_image_formats type
        ushort height, width, colors, bits
        unsigned int  data_size 
        unsigned char data[1] # this is the image data, no idea why [1]

# The open_file method is overloaded on Windows and unfortunately
# there is no better way to deal with this in Cython.
IF UNAME_SYSNAME == "Windows":
    cdef extern from "libraw.h":
        cdef cppclass LibRaw:
            libraw_data_t imgdata
            LibRaw()
            int open_buffer(void *buffer, size_t bufsize) nogil
            int open_file(const wchar_t *fname) nogil
            int unpack() nogil
            int unpack_thumb() nogil
            int COLOR(int row, int col) nogil
            int error_count() nogil
            int dcraw_process() nogil
            libraw_processed_image_t* dcraw_make_mem_image(int *errcode) nogil
            libraw_processed_image_t* dcraw_make_mem_thumb(int *errcode) nogil
            void dcraw_clear_mem(libraw_processed_image_t* img) nogil
            void free_image() nogil
            const char* strerror(int p) nogil
            void recycle() nogil
ELSE:
    cdef extern from "libraw.h":
        cdef cppclass LibRaw:
            libraw_data_t imgdata
            LibRaw()
            int open_buffer(void *buffer, size_t bufsize) nogil
            int open_file(const char *fname) nogil
            int unpack() nogil
            int unpack_thumb() nogil
            int COLOR(int row, int col)
            int error_count() nogil
            int dcraw_process() nogil
            libraw_processed_image_t* dcraw_make_mem_image(int *errcode) nogil
            libraw_processed_image_t* dcraw_make_mem_thumb(int *errcode) nogil
            void dcraw_clear_mem(libraw_processed_image_t* img) nogil
            void free_image() nogil
            const char* strerror(int p) nogil
            void recycle() nogil

libraw_version = (LIBRAW_MAJOR_VERSION, LIBRAW_MINOR_VERSION, LIBRAW_PATCH_VERSION)

if _LIBRAW_HAS_FLAGS:
    flags = {'DNGLOSSYCODEC': bool(_LIBRAW_USE_DNGLOSSYCODEC),
             'DNGDEFLATECODEC': bool(_LIBRAW_USE_DNGDEFLATECODEC),
             'OPENMP': bool(_LIBRAW_USE_OPENMP),
             'LCMS': bool(_LIBRAW_USE_LCMS),
             'REDCINECODEC': bool(_LIBRAW_USE_REDCINECODEC),
             'RAWSPEED': bool(_LIBRAW_USE_RAWSPEED),
             'DEMOSAIC_PACK_GPL2': bool(_LIBRAW_USE_DEMOSAIC_PACK_GPL2),
             'DEMOSAIC_PACK_GPL3': bool(_LIBRAW_USE_DEMOSAIC_PACK_GPL3),
             'X3FTOOLS': bool(_LIBRAW_USE_X3FTOOLS),
             '6BY9RPI': bool(_LIBRAW_USE_6BY9RPI),
             }
else:
    flags = None

ImageSizes = namedtuple('ImageSizes', ['raw_height', 'raw_width', 
                                       'height', 'width', 
                                       'top_margin', 'left_margin',
                                       'iheight', 'iwidth',
                                       'pixel_aspect', 'flip',
                                       'crop_left_margin', 'crop_top_margin', 'crop_width', 'crop_height'
                                       ])

class RawType(Enum):
    """
    RAW image type.
    """
    
    Flat = 0
    """ Bayer type or black and white """
    
    Stack = 1
    """ Foveon type or sRAW/mRAW files or RawSpeed decoding """

# LibRaw_thumbnail_formats
class ThumbFormat(Enum):
    """
    Thumbnail/preview image type.
    """

    JPEG = 1
    """ JPEG image as bytes object. """

    BITMAP = 2
    """ RGB image as ndarray object. """

Thumbnail = namedtuple('Thumbnail', ['format', 'data'])

class LibRawError(Exception):
    pass

class LibRawFatalError(LibRawError):
    pass

class LibRawNonFatalError(LibRawError):
    pass

class LibRawUnspecifiedError(LibRawNonFatalError):
    pass

class LibRawFileUnsupportedError(LibRawNonFatalError):
    pass

class LibRawRequestForNonexistentImageError(LibRawNonFatalError):
    pass

class LibRawOutOfOrderCallError(LibRawNonFatalError):
    pass

class LibRawNoThumbnailError(LibRawNonFatalError):
    pass

class LibRawUnsupportedThumbnailError(LibRawNonFatalError):
    pass

class LibRawInputClosedError(LibRawNonFatalError):
    pass

class LibRawNotImplementedError(LibRawNonFatalError):
    pass

class LibRawUnsufficientMemoryError(LibRawFatalError):
    pass

class LibRawDataError(LibRawFatalError):
    pass

class LibRawIOError(LibRawFatalError):
    pass

class LibRawCancelledByCallbackError(LibRawFatalError):
    pass

class LibRawBadCropError(LibRawFatalError):
    pass

class LibRawTooBigError(LibRawFatalError):
    pass

class LibRawMemPoolOverflowError(LibRawFatalError):
    pass

# From LibRaw_errors in libraw_const.h
_LIBRAW_ERROR_MAP = {
    -1: LibRawUnspecifiedError,
    -2: LibRawFileUnsupportedError,
    -3: LibRawRequestForNonexistentImageError,
    -4: LibRawOutOfOrderCallError,
    -5: LibRawNoThumbnailError,
    -6: LibRawUnsupportedThumbnailError,
    -7: LibRawInputClosedError,
    -8: LibRawNotImplementedError,
    -100007: LibRawUnsufficientMemoryError,
    -100008: LibRawDataError,
    -100009: LibRawIOError,
    -100010: LibRawCancelledByCallbackError,
    -100011: LibRawBadCropError,
    -100012: LibRawTooBigError,
    -100013: LibRawMemPoolOverflowError
}

cdef class RawPy:
    """
    Load RAW images, work on their data, and create a postprocessed (demosaiced) image.
    
    All operations are implemented using numpy arrays.
    """
    cdef LibRaw* p
    cdef bint unpack_called
    cdef bint unpack_thumb_called
    cdef bint dcraw_process_called
    cdef object bytes
        
    def __cinit__(self):
        self.unpack_called = False
        self.unpack_thumb_called = False
        self.dcraw_process_called = False
        self.p = new LibRaw()
        
    def __dealloc__(self):
        del self.p
        
    def __enter__(self) -> RawPy:
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
        
    def close(self) -> None:
        """
        Release all resources and close the RAW image.
        
        Consider using context managers for the same effect:
        
        .. code-block:: python
        
            with rawpy.imread('image.nef') as raw:
              # work with raw object
        
        """
        with nogil:
            self.p.recycle()
    
    def open_file(self, path: str) -> None:
        """
        Opens the given RAW image file. Should be followed by a call to :meth:`~rawpy.RawPy.unpack`.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        
        :param str path: The path to the RAW image.
        """
        cdef wchar_t *wchars
        cdef Py_ssize_t wchars_len
        self.unpack_called = False
        self.unpack_thumb_called = False
        self.dcraw_process_called = False
        IF UNAME_SYSNAME == "Windows":
            wchars = PyUnicode_AsWideCharString(path, &wchars_len)
            if wchars == NULL:
                raise RuntimeError('cannot convert unicode path to wide chars')
            with nogil:
                res = self.p.open_file(wchars)
            PyMem_Free(wchars)
        ELSE:
            res = self.p.open_file(path.encode('UTF-8'))
        self.handle_error(res)
    
    def open_buffer(self, fileobj: BinaryIO) -> None:
        """
        Opens the given RAW image file-like object. Should be followed by a call to :meth:`~rawpy.RawPy.unpack`.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        
        :param file fileobj: The file-like object.
        """
        self.unpack_called = False
        self.unpack_thumb_called = False
        self.dcraw_process_called = False
        # we keep a reference to the byte buffer to avoid garbage collection
        self.bytes = fileobj.read()
        cdef char *buf = self.bytes
        buf_len = len(self.bytes)
        with nogil:
            e = self.p.open_buffer(buf, buf_len)
        self.handle_error(e)
    
    def set_unpack_params(self, shot_select: int = 0) -> None:
        """
        Set parameters that affect RAW image unpacking.
        
        This should be called after opening a file and before unpacking.
        
        .. NOTE:: This is a low-level method. When using :func:`rawpy.imread`,
                  unpack parameters can be provided directly.
        
        :param int shot_select: select which image to extract from RAW files that contain multiple images
                                (e.g., Dual Pixel RAW). Default is 0 for the first/main image.
        """
        cdef libraw_raw_unpack_params_t* rp = &self.p.imgdata.rawparams
        rp.shot_select = shot_select
    
    def unpack(self) -> None:
        """
        Unpacks/decodes the opened RAW image.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        """
        with nogil:
            e = self.p.unpack()
        self.handle_error(e)
        self.bytes = None
        self.unpack_called = True

    cdef ensure_unpack(self):
        if not self.unpack_called:
            self.unpack()

    def unpack_thumb(self) -> None:
        """
        Unpacks/decodes the thumbnail/preview image, whichever is bigger.
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.extract_thumb` instead.
        """
        with nogil:
            e = self.p.unpack_thumb()
        self.handle_error(e)
        self.unpack_thumb_called = True

    cdef ensure_unpack_thumb(self):
        if not self.unpack_thumb_called:
            self.unpack_thumb()
    
    @property
    def raw_type(self) -> RawType:
        """
        Return the RAW type.
        
        :rtype: :class:`rawpy.RawType`
        """
        self.ensure_unpack()
        if self.p.imgdata.rawdata.raw_image != NULL:
            return RawType.Flat
        else:
            return RawType.Stack
    
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
        self.ensure_unpack()
        cdef np.npy_intp shape_bayer[2]
        cdef np.npy_intp shape_rgb[3]
        cdef np.ndarray ndarr
        if self.p.imgdata.rawdata.raw_image != NULL:
            shape_bayer[0] = <np.npy_intp> self.p.imgdata.sizes.raw_height
            shape_bayer[1] = <np.npy_intp> self.p.imgdata.sizes.raw_width
            ndarr = np.PyArray_SimpleNewFromData(2, shape_bayer, np.NPY_USHORT, self.p.imgdata.rawdata.raw_image)
        elif self.p.imgdata.rawdata.color3_image != NULL:
            shape_rgb[0] = <np.npy_intp> self.p.imgdata.sizes.raw_height
            shape_rgb[1] = <np.npy_intp> self.p.imgdata.sizes.raw_width
            shape_rgb[2] = <np.npy_intp> 3
            ndarr = np.PyArray_SimpleNewFromData(3, shape_rgb, np.NPY_USHORT, self.p.imgdata.rawdata.color3_image)
        elif self.p.imgdata.rawdata.color4_image != NULL:
            shape_rgb[0] = <np.npy_intp> self.p.imgdata.sizes.raw_height
            shape_rgb[1] = <np.npy_intp> self.p.imgdata.sizes.raw_width
            shape_rgb[2] = <np.npy_intp> 4
            ndarr = np.PyArray_SimpleNewFromData(3, shape_rgb, np.NPY_USHORT, self.p.imgdata.rawdata.color4_image)
        else:
            raise RuntimeError('unsupported raw data')

        # ndarr must hold a reference to this object,
        # otherwise the underlying data gets lost when the RawPy instance gets out of scope
        # (which would trigger __dealloc__)
        np.PyArray_SetBaseObject(ndarr, self)
        # Python doesn't know about above assignment as it's in C-level 
        Py_INCREF(self)
        return ndarr
    
    @property
    def raw_image_visible(self) -> NDArray[np.uint16]:
        """
        Like raw_image but without margin.
        
        :rtype: ndarray of shape (hv,wv[,c])
        """
        self.ensure_unpack()
        s = self.sizes
        return self.raw_image[s.top_margin:s.top_margin+s.height,
                              s.left_margin:s.left_margin+s.width]
            
    cpdef ushort raw_value(self, int row, int column):
        """
        Return RAW value at given position relative to the full RAW image.
        Only usable for flat RAW images (see :attr:`~rawpy.RawPy.raw_type` property).
        """
        self.ensure_unpack()
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        if raw == NULL:
            raise RuntimeError('RAW image is not flat')
        cdef ushort raw_width = self.p.imgdata.sizes.raw_width
        return raw[row*raw_width + column]
            
    cpdef ushort raw_value_visible(self, int row, int column):
        """
        Return RAW value at given position relative to visible area of image.
        Only usable for flat RAW images (see :attr:`~rawpy.RawPy.raw_type` property).        
        """
        self.ensure_unpack()
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        if raw == NULL:
            raise RuntimeError('RAW image is not flat')
        cdef ushort top_margin = self.p.imgdata.sizes.top_margin
        cdef ushort left_margin = self.p.imgdata.sizes.left_margin
        cdef ushort raw_width = self.p.imgdata.sizes.raw_width
        return raw[(row+top_margin)*raw_width + column + left_margin]
        
    @property
    def sizes(self) -> ImageSizes:
        """
        Return a :class:`rawpy.ImageSizes` instance with size information of
        the RAW image and postprocessed image.        
        """
        cdef libraw_image_sizes_t* s = &self.p.imgdata.sizes

        # LibRaw returns 65535 for cleft and ctop in some files - probably those that do not specify them
        cdef bint has_cleft = s.raw_inset_crops[0].cleft != USHRT_MAX
        cdef bint has_ctop = s.raw_inset_crops[0].ctop != USHRT_MAX

        return ImageSizes(raw_height=s.raw_height, raw_width=s.raw_width,
                          height=s.height, width=s.width,
                          top_margin=s.top_margin, left_margin=s.left_margin,
                          iheight=s.iheight, iwidth=s.iwidth,
                          pixel_aspect=s.pixel_aspect, flip=s.flip,
                          crop_left_margin=s.raw_inset_crops[0].cleft if has_cleft else 0,
                          crop_top_margin=s.raw_inset_crops[0].ctop if has_ctop else 0,
                          crop_width=s.raw_inset_crops[0].cwidth, crop_height=s.raw_inset_crops[0].cheight)
    
    @property
    def num_colors(self) -> int:
        """
        Number of colors.
        Note that e.g. for RGBG this can be 3 or 4, depending on the camera model,
        as some use two different greens. 
        """
        return self.p.imgdata.idata.colors
    
    @property
    def color_desc(self) -> bytes:
        """
        String description of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG).
        Note that same letters may not refer strictly to the same color.
        There are cameras with two different greens for example.
        """
        return self.p.imgdata.idata.cdesc
    
    cpdef int raw_color(self, int row, int column):
        """
        Return color index for the given coordinates relative to the full RAW size.
        Only usable for flat RAW images (see raw_type property).
        """
        self.ensure_unpack()
        if self.p.imgdata.rawdata.raw_image == NULL:
            raise RuntimeError('RAW image is not flat')
        cdef ushort top_margin = self.p.imgdata.sizes.top_margin
        cdef ushort left_margin = self.p.imgdata.sizes.left_margin
        # COLOR's coordinates are relative to visible image size.
        return self.p.COLOR(row - top_margin, column - left_margin)
    
    @property
    def raw_colors(self) -> NDArray[np.uint8]:
        """
        An array of color indices for each pixel in the RAW image.
        Equivalent to calling raw_color(y,x) for each pixel.
        Only usable for flat RAW images (see raw_type property).
        
        :rtype: ndarray of shape (h,w)
        """
        self.ensure_unpack()
        if self.p.imgdata.rawdata.raw_image == NULL:
            raise RuntimeError('RAW image is not flat')
        cdef np.ndarray pattern = self.raw_pattern
        cdef int n = pattern.shape[0]
        cdef int height = self.p.imgdata.sizes.raw_height
        cdef int width = self.p.imgdata.sizes.raw_width
        return np.pad(pattern, ((0, height - n), (0, width - n)), mode='wrap')
    
    @property
    def raw_colors_visible(self) -> NDArray[np.uint8]:
        """
        Like raw_colors but without margin.
        
        :rtype: ndarray of shape (hv,wv)
        """
        s = self.sizes
        return self.raw_colors[s.top_margin:s.top_margin+s.height,
                               s.left_margin:s.left_margin+s.width]
    
    @property
    def raw_pattern(self) -> Optional[NDArray[np.uint8]]:
        """
        The smallest possible Bayer pattern of this image.
        
        :rtype: ndarray, or None if not a flat RAW image
        """
        self.ensure_unpack()
        if self.p.imgdata.rawdata.raw_image == NULL:
            return None
        cdef np.ndarray pattern
        cdef int n
        if self.p.imgdata.idata.filters < 1000:
            if self.p.imgdata.idata.filters == 0:
                # black and white
                n = 1
            elif self.p.imgdata.idata.filters == 1:
                # Leaf Catchlight
                n = 16
            elif self.p.imgdata.idata.filters == LIBRAW_XTRANS:
                n = 6
            else:
                raise NotImplementedError('filters: {}'.format(self.p.imgdata.idata.filters))
        else:
            n = 4
        
        pattern = np.empty((n, n), dtype=np.uint8)
        cdef int y, x
        for y in range(n):
            for x in range(n):
                pattern[y,x] = self.raw_color(y, x)
        if n == 4:
            if np.all(pattern[:2,:2] == pattern[:2,2:]) and \
               np.all(pattern[:2,:2] == pattern[2:,2:]) and \
               np.all(pattern[:2,:2] == pattern[2:,:2]):
                pattern = pattern[:2,:2]
        return pattern
       
    @property
    def camera_whitebalance(self) -> List[float]:
        """
        White balance coefficients (as shot). Either read from file or calculated.
        
        :rtype: list of length 4
        """
        self.ensure_unpack()
        return [self.p.imgdata.rawdata.color.cam_mul[0],
                self.p.imgdata.rawdata.color.cam_mul[1],
                self.p.imgdata.rawdata.color.cam_mul[2],
                self.p.imgdata.rawdata.color.cam_mul[3]]
        
    @property
    def daylight_whitebalance(self) -> List[float]:
        """
        White balance coefficients for daylight (daylight balance). 
        Either read from file, or calculated on the basis of file data, 
        or taken from hardcoded constants.
        
        :rtype: list of length 4
        """
        self.ensure_unpack()
        return [self.p.imgdata.rawdata.color.pre_mul[0],
                self.p.imgdata.rawdata.color.pre_mul[1],
                self.p.imgdata.rawdata.color.pre_mul[2],
                self.p.imgdata.rawdata.color.pre_mul[3]]
    
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
        self.ensure_unpack()
        if not self.dcraw_process_called:
            return None
        return [self.p.imgdata.color.pre_mul[0],
                self.p.imgdata.color.pre_mul[1],
                self.p.imgdata.color.pre_mul[2],
                self.p.imgdata.color.pre_mul[3]]
    
    @property
    def black_level_per_channel(self) -> List[int]:
        """
        Per-channel black level correction.
        
        :rtype: list of length 4
        """
        self.ensure_unpack()
        cdef libraw_colordata_black_level_t bl = adjust_bl_(self.p)
        return [bl.cblack[0],
                bl.cblack[1],
                bl.cblack[2],
                bl.cblack[3]]

    @property
    def white_level(self) -> int:
        """
        Level at which the raw pixel value is considered to be saturated.
        """
        self.ensure_unpack()
        return self.p.imgdata.rawdata.color.maximum

    @property
    def camera_white_level_per_channel(self) -> Optional[List[int]]:
        """
        Per-channel saturation levels read from raw file metadata, if it exists. Otherwise None.

        :rtype: list of length 4, or None if metadata missing
        """
        self.ensure_unpack()
        levels = [self.p.imgdata.rawdata.color.linear_max[0],
                  self.p.imgdata.rawdata.color.linear_max[1],
                  self.p.imgdata.rawdata.color.linear_max[2],
                  self.p.imgdata.rawdata.color.linear_max[3]]
        if all(l > 0 for l in levels):
            return levels
        else:
            return None

    @property
    def color_matrix(self) -> NDArray[np.float32]:
        """
        Color matrix, read from file for some cameras, calculated for others. 
        
        :rtype: ndarray of shape (3,4)
        """
        self.ensure_unpack()
        cdef np.ndarray matrix = np.empty((3, 4), dtype=np.float32)
        for i in range(3):
            for j in range(4):
                matrix[i,j] = self.p.imgdata.rawdata.color.cmatrix[i][j]
        return matrix
        
    @property
    def rgb_xyz_matrix(self) -> NDArray[np.float32]:
        """
        Camera RGB - XYZ conversion matrix.
        This matrix is constant (different for different models).
        Last row is zero for RGB cameras and non-zero for different color models (CMYG and so on).
        
        :rtype: ndarray of shape (4,3)
        """
        self.ensure_unpack()
        cdef np.ndarray matrix = np.empty((4, 3), dtype=np.float32)
        for i in range(4):
            for j in range(3):
                matrix[i,j] = self.p.imgdata.rawdata.color.cam_xyz[i][j]
        return matrix
    
    @property
    def tone_curve(self) -> NDArray[np.uint16]:
        """
        Camera tone curve, read from file for Nikon, Sony and some other cameras.
        
        :rtype: ndarray of length 65536
        """
        self.ensure_unpack()
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 65536
        return np.PyArray_SimpleNewFromData(1, shape, np.NPY_USHORT,
                                            &self.p.imgdata.rawdata.color.curve)

    def dcraw_process(self, params: Optional[Params] = None, **kw) -> None:
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
        self.ensure_unpack()
        if params and kw:
            raise ValueError('If params is given, then no additional keywords are allowed')
        if params is None:
            params = Params(**kw)
        self.apply_params(params)
        with nogil:
            e = self.p.dcraw_process()
        self.handle_error(e)
        self.dcraw_process_called = True
        
    def dcraw_make_mem_image(self):
        """
        Return the postprocessed image (see :meth:`~rawpy.RawPy.dcraw_process`) as numpy array.
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.postprocess` instead.
        
        :rtype: ndarray of shape (h,w,c)
        """
        cdef int errcode = 0
        cdef libraw_processed_image_t* img
        with nogil:
            img = self.p.dcraw_make_mem_image(&errcode)
        self.handle_error(errcode)
        assert img.type == LIBRAW_IMAGE_BITMAP
        wrapped = processed_image_wrapper()
        wrapped.set_data(self, img)
        ndarr = wrapped.__array__()
        return ndarr

    def dcraw_make_mem_thumb(self):
        """
        Return the thumbnail/preview image (see :meth:`~rawpy.RawPy.unpack_thumb`)
        as :class:`rawpy.Thumbnail` object.
        For JPEG thumbnails, data is a bytes object and can be written as-is to file.
        For bitmap thumbnails, data is an ndarray of shape (h,w,c).
        If no image exists or the format is unsupported, an exception is raised.
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.extract_thumb` instead.
        
        :rtype: :class:`rawpy.Thumbnail`
        """
        cdef int errcode = 0
        cdef libraw_processed_image_t* img
        with nogil:
            img = self.p.dcraw_make_mem_thumb(&errcode)
        self.handle_error(errcode)
        if img.type == LIBRAW_IMAGE_BITMAP:
            wrapped = processed_image_wrapper()
            wrapped.set_data(self, img)
            data = wrapped.__array__()
            return Thumbnail(ThumbFormat.BITMAP, data)
        elif img.type == LIBRAW_IMAGE_JPEG:
            # Note: This creates a copy.
            data = PyBytes_FromStringAndSize(<char*>img.data, img.data_size)
            self.p.dcraw_clear_mem(img)
            return Thumbnail(ThumbFormat.JPEG, data)
        else:
            raise NotImplementedError('thumb type: {}'.format(img.type))

    def extract_thumb(self) -> Thumbnail:
        """
        Extracts and returns the thumbnail/preview image (whichever is bigger)
        of the opened RAW image as :class:`rawpy.Thumbnail` object.
        For JPEG thumbnails, data is a bytes object and can be written as-is to file.
        For bitmap thumbnails, data is an ndarray of shape (h,w,c).
        If no image exists or the format is unsupported, an exception is raised.

        .. code-block:: python
            import imageio.v3 as iio

            ...
        
            with rawpy.imread('image.nef') as raw:
              try:
                thumb = raw.extract_thumb()
              except rawpy.LibRawNoThumbnailError:
                print('no thumbnail found')
              except rawpy.LibRawUnsupportedThumbnailError:
                print('unsupported thumbnail')
              else:
                if thumb.format == rawpy.ThumbFormat.JPEG:
                  with open('thumb.jpg', 'wb') as f:
                    f.write(thumb.data)
                elif thumb.format == rawpy.ThumbFormat.BITMAP:
                  iio.imwrite('thumb.tiff', thumb.data)
        
        :rtype: :class:`rawpy.Thumbnail`
        """
        self.ensure_unpack_thumb()
        thumb = self.dcraw_make_mem_thumb()
        return thumb
    
    def postprocess(self, params: Optional[Params] = None, **kw) -> NDArray[np.uint8]:
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
        self.dcraw_process(params, **kw)
        return self.dcraw_make_mem_image()
        
    cdef apply_params(self, params):
        if params is None:
            return
        cdef libraw_output_params_t* p = &self.p.imgdata.params
        p.user_qual = params.user_qual
        p.half_size = params.half_size
        p.four_color_rgb = params.four_color_rgb
        p.dcb_iterations = params.dcb_iterations
        p.dcb_enhance_fl = params.dcb_enhance_fl
        p.fbdd_noiserd = params.fbdd_noiserd
        p.threshold = params.threshold
        p.med_passes = params.med_passes
        p.use_camera_wb = params.use_camera_wb
        p.use_auto_wb = params.use_auto_wb
        if params.user_mul:
            for i in range(4):
                p.user_mul[i] = params.user_mul[i]
        p.output_color = params.output_color
        p.output_bps = params.output_bps
        p.user_flip = params.user_flip
        p.user_black = params.user_black
        if params.user_cblack:
            for i in range(4):
                p.user_cblack[i] = params.user_cblack[i]
        p.user_sat = params.user_sat
        p.no_auto_bright = params.no_auto_bright
        p.no_auto_scale = params.no_auto_scale
        p.auto_bright_thr = params.auto_bright_thr
        p.adjust_maximum_thr = params.adjust_maximum_thr
        p.bright = params.bright
        p.highlight = params.highlight
        p.exp_correc = params.exp_correc
        p.exp_shift = params.exp_shift
        p.exp_preser = params.exp_preser
        if params.bad_pixels:
            p.bad_pixels = params.bad_pixels
        else:
            p.bad_pixels = NULL
        p.gamm[0] = params.gamm[0]
        p.gamm[1] = params.gamm[1]
        p.aber[0] = params.aber[0]
        p.aber[2] = params.aber[1]
    
    cdef handle_error(self, int code):
        if code > 0:
            raise OSError((code, os.strerror(code))) 
        elif code < 0:
            errstr = self.p.strerror(code)
            if code in _LIBRAW_ERROR_MAP:
                raise _LIBRAW_ERROR_MAP[code](errstr)
            elif code < -10000: # see macro LIBRAW_FATAL_ERROR in libraw_const.h
                raise LibRawFatalError(errstr)
            else:
                raise LibRawNonFatalError(errstr)

        cdef int error_count
        with nogil:
            error_count = self.p.error_count()
        if error_count > 0:
            raise LibRawDataError("Data error or unsupported file format")

class DemosaicAlgorithm(Enum):
    """
    Identifiers for demosaic algorithms.
    """
    LINEAR=0
    VNG=1
    PPG=2
    AHD=3
    DCB=4
    # 5-9 only usable if demosaic pack GPL2 available
    MODIFIED_AHD=5
    AFD=6
    VCD=7
    VCD_MODIFIED_AHD=8
    LMMSE=9
    # 10 only usable if demosaic pack GPL3 available
    AMAZE=10
    # 11-12 only usable for LibRaw >= 0.16
    DHT=11
    AAHD=12
    
    @property
    def isSupported(self):
        """
        Return True if the demosaic algorithm is supported, False if it is not,
        and None if the support status is unknown. The latter is returned if
        LibRaw < 0.15.4 is used or if it was compiled without cmake.
        
        The necessary information is read from the libraw_config.h header which
        is only written with cmake builds >= 0.15.4.
        """
        try:
            supported = self.checkSupported()
        except NotSupportedError:
            return False
        else:
            return supported
        
    def checkSupported(self):
        """
        Like :attr:`isSupported` but raises an exception for the `False` case.
        """
        c = DemosaicAlgorithm
        
        min_version_flags = (0,15,4)
        min_version_dht_aahd = (0,16,0)
       
        if self in [c.MODIFIED_AHD, c.AFD, c.VCD, c.VCD_MODIFIED_AHD, c.LMMSE]:
            if flags is None:
                return None
            elif not _LIBRAW_USE_DEMOSAIC_PACK_GPL2:
                raise NotSupportedError('Demosaic algorithm ' + self.name + ' requires GPL2 demosaic pack')
            
        elif self in [c.AMAZE]:
            if flags is None:
                return None
            elif not _LIBRAW_USE_DEMOSAIC_PACK_GPL3:
                raise NotSupportedError('Demosaic algorithm ' + self.name + ' requires GPL3 demosaic pack')
        
        elif self in [c.DHT, c.AAHD] and \
           libraw_version < min_version_dht_aahd:
            raise NotSupportedError('Demosaic algorithm ' + self.name, min_version_dht_aahd)
        return True

class FBDDNoiseReductionMode(Enum):
    """
    FBDD noise reduction modes.
    """
    Off=0
    Light=1
    Full=2

class ColorSpace(Enum):
    """
    Color spaces.
    """
    raw=0
    sRGB=1
    Adobe=2
    Wide=3
    ProPhoto=4
    XYZ=5
    ACES=6
    P3D65=7
    Rec2020=8
    
class HighlightMode(Enum):
    """
    Highlight modes.
    """
    Clip=0
    Ignore=1
    Blend=2
    ReconstructDefault=5
    
    @classmethod
    def Reconstruct(self, level):
        """
        :param int level: 3 to 9, low numbers favor whites, high numbers favor colors
        """
        if not 3 <= level <= 9:
            raise ValueError('highlight reconstruction level must be between 3 and 9 inclusive') 
        return level


class NotSupportedError(Exception):
    def __init__(self, message, min_version=None):
        if min_version is not None:
            message = "{}, minimum required LibRaw version: {}.{}.{}, your version: {}.{}.{}".format(
                          message, min_version[0], min_version[1], min_version[2],
                          libraw_version[0], libraw_version[1], libraw_version[2])
        Exception.__init__(self, message)

class Params(object):
    """
    A class that handles postprocessing parameters.
    """
    def __init__(self, 
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
                 bad_pixels_path: Optional[str] = None) -> None:
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

        if demosaic_algorithm:
            demosaic_algorithm.checkSupported()
            self.user_qual = demosaic_algorithm.value
        else:
            self.user_qual = -1
        self.half_size = half_size
        self.four_color_rgb = four_color_rgb
        self.dcb_iterations = dcb_iterations
        self.dcb_enhance_fl = dcb_enhance
        self.fbdd_noiserd = fbdd_noise_reduction.value
        self.threshold = noise_thr if noise_thr is not None else 0.0
        self.med_passes = median_filter_passes
        self.use_camera_wb = use_camera_wb
        self.use_auto_wb = use_auto_wb
        if user_wb is not None:
            assert len(user_wb) == 4
            self.user_mul = user_wb
        else:
            self.user_mul = [0,0,0,0] 
        self.output_color = output_color.value
        self.output_bps = output_bps
        self.user_flip = user_flip if user_flip is not None else -1
        self.user_black = user_black if user_black is not None else -1
        if user_cblack is not None:
            assert len(user_cblack) == 4
            self.user_cblack = user_cblack
        else:
            self.user_cblack = None
        self.user_sat = user_sat if user_sat is not None else -1
        self.no_auto_bright = no_auto_bright
        self.no_auto_scale = no_auto_scale
        if auto_bright_thr is not None:
            min_version = (0,16,1)
            if libraw_version < min_version:
                # see https://github.com/LibRaw/LibRaw/commit/ea70421a518ba5a039fcc1dc1045b428159fb032
                raise NotSupportedError('Parameter auto_bright_thr', min_version)
            self.auto_bright_thr = auto_bright_thr
        else:
            self.auto_bright_thr = LIBRAW_DEFAULT_AUTO_BRIGHTNESS_THRESHOLD
        self.adjust_maximum_thr = adjust_maximum_thr
        self.bright = bright
        if isinstance(highlight_mode, HighlightMode):
            self.highlight = highlight_mode.value
        else:
            self.highlight = highlight_mode
        if exp_shift is not None:
            self.exp_correc = 1
            self.exp_shift = exp_shift
        else:
            self.exp_correc = -1
            self.exp_shift = 1.0
        self.exp_preser = exp_preserve_highlights
        if gamma is not None:
            assert len(gamma) == 2
            self.gamm = (1/gamma[0], gamma[1])
        else:
            self.gamm = (1/2.222, 4.5) # rec. BT.709
        if chromatic_aberration is not None:
            assert len(chromatic_aberration) == 2
            self.aber = (chromatic_aberration[0], chromatic_aberration[1])
        else:
            self.aber = (1, 1)
        self.bad_pixels = bad_pixels_path
    
cdef class processed_image_wrapper:
    cdef RawPy raw
    cdef libraw_processed_image_t* processed_image

    cdef set_data(self, RawPy raw, libraw_processed_image_t* processed_image):
        self.raw = raw
        self.processed_image = processed_image

    def __array__(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.processed_image.height
        shape[1] = <np.npy_intp> self.processed_image.width
        shape[2] = <np.npy_intp> self.processed_image.colors
        cdef np.ndarray ndarr
        ndarr = np.PyArray_SimpleNewFromData(3, shape, 
                                             np.NPY_UINT8 if self.processed_image.bits == 8 else np.NPY_UINT16,
                                             self.processed_image.data)
        np.PyArray_SetBaseObject(ndarr, self)
        # Python doesn't know about above assignment as it's in C-level 
        Py_INCREF(self)
        return ndarr

    def __dealloc__(self):
        self.raw.p.dcraw_clear_mem(self.processed_image)        
    
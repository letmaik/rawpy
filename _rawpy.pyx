# distutils: language = c++
# cython: embedsignature=True

from __future__ import print_function

from cpython.ref cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref

import numpy as np
from collections import namedtuple
cimport numpy as np
np.import_array()

import os
import sys
import warnings
from enum import Enum

cdef extern from "def_helper.h":
    cdef int LIBRAW_XTRANS
    
    cdef int _LIBRAW_HAS_FLAGS
    # the following flags are only usable if _LIBRAW_HAS_FLAGS is 1
    # (this is the case for libraw >= 0.15.4 and only when cmake was used)
    cdef int _LIBRAW_USE_DNGLOSSYCODEC
    cdef int _LIBRAW_USE_OPENMP
    cdef int _LIBRAW_USE_LCMS
    cdef int _LIBRAW_USE_REDCINECODEC
    cdef int _LIBRAW_USE_RAWSPEED
    cdef int _LIBRAW_USE_DEMOSAIC_PACK_GPL2
    cdef int _LIBRAW_USE_DEMOSAIC_PACK_GPL3

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
    
    ctypedef struct libraw_image_sizes_t:
        ushort raw_height, raw_width
        ushort height, width
        ushort top_margin, left_margin
        ushort iheight, iwidth
        double pixel_aspect
        int flip
        
    ctypedef struct libraw_colordata_t:
        float       cam_mul[4] 
        float       pre_mul[4]
        ushort      curve[0x10000] # 65536
        unsigned    cblack[4]
        unsigned    black
        float       cmatrix[3][4]
        float       cam_xyz[4][3]
        void        *profile # a string?
        unsigned    profile_length

    ctypedef struct libraw_rawdata_t:
        ushort *raw_image # 1 component per pixel, for b/w and Bayer type sensors
        # color4_image and color3_image supported since 0.15
        # There is no easy way to include these conditionally, so for now (as we support 0.14)
        # we don't support them.
        #ushort        (*color4_image)[4] # 4 components per pixel, the 4th component can be void
        #ushort        (*color3_image)[3] # 3 components per pixel, sRAW/mRAW files, RawSpeed decoding
        libraw_colordata_t          color
        
    ctypedef struct libraw_output_params_t:
        unsigned    greybox[4]     # -A  x1 y1 x2 y2 
        unsigned    cropbox[4]     # -B x1 y1 x2 y2 
        double      aber[4]        # -C 
        double      gamm[6]        # -g 
        float       user_mul[4]    # -r mul0 mul1 mul2 mul3 
        unsigned    shot_select    # -s 
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

    cdef cppclass LibRaw:
        libraw_data_t imgdata
        LibRaw()
        int open_file(const char *fname)
        int unpack()
        int COLOR(int row, int col)
#         int raw2image()
        int dcraw_process()
        libraw_processed_image_t* dcraw_make_mem_image(int *errcode)
        void dcraw_clear_mem(libraw_processed_image_t* img)
        void free_image()
        const char* strerror(int p)
        void recycle()

libraw_version = (LIBRAW_MAJOR_VERSION, LIBRAW_MINOR_VERSION, LIBRAW_PATCH_VERSION)

if _LIBRAW_HAS_FLAGS:
    flags = {'DNGLOSSYCODEC': bool(_LIBRAW_USE_DNGLOSSYCODEC),
             'OPENMP': bool(_LIBRAW_USE_OPENMP),
             'LCMS': bool(_LIBRAW_USE_LCMS),
             'REDCINECODEC': bool(_LIBRAW_USE_REDCINECODEC),
             'RAWSPEED': bool(_LIBRAW_USE_RAWSPEED),
             'DEMOSAIC_PACK_GPL2': bool(_LIBRAW_USE_DEMOSAIC_PACK_GPL2),
             'DEMOSAIC_PACK_GPL3': bool(_LIBRAW_USE_DEMOSAIC_PACK_GPL3),
             }
else:
    flags = None

ImageSizes = namedtuple('ImageSizes', ['raw_height', 'raw_width', 
                                       'height', 'width', 
                                       'top_margin', 'left_margin',
                                       'iheight', 'iwidth',
                                       'pixel_aspect', 'flip'])

class RawType(Enum):
    Flat = 0
    """ Bayer type or black and white """
    
    Stack = 1
    """ Foveon type or sRAW/mRAW files or RawSpeed decoding """

cdef class RawPy:
    """
    Load RAW images, work on their data, and create a postprocessed (demosaiced) image.
    
    All operations are implemented using numpy arrays.
    """
    cdef LibRaw* p
    cdef bint needs_reopening
        
    def __cinit__(self):
        self.p = new LibRaw()
        
    def __dealloc__(self):
        del self.p
    
    def open_file(self, path):
        """
        Opens the given RAW image file. Should be followed by a call to :meth:`~rawpy.RawPy.unpack`.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        
        :param str path: The path to the RAW image.
        """
        self.handle_error(self.p.open_file(_chars(path)))
        if libraw_version < (0,15):
            # libraw < 0.15 requires calling open_file & unpack for multiple calls to dcraw_process
            # with different parameters, therefore we remember the fact that this is freshly opened
            # and issue a warning in postprocess if needed
            self.needs_reopening = False
        
    def unpack(self):
        """
        Unpacks/decodes the opened RAW image.
        
        .. NOTE:: This is a low-level method, consider using :func:`rawpy.imread` instead.
        """
        self.handle_error(self.p.unpack())
    
    property raw_type:
        """
        Return the RAW type.
        
        :rtype: :class:`rawpy.RawType`
        """
        def __get__(self):
            if self.p.imgdata.rawdata.raw_image != NULL:
                return RawType.Flat
            else:
                return RawType.Stack
    
    property raw_image:
        """
        View of Bayer-pattern RAW image, one channel. Includes margin.
        
        :rtype: ndarray of shape (h,w)
        """
        def __get__(self):
            cdef ushort* raw = self.p.imgdata.rawdata.raw_image
            if raw == NULL:
                raise NotImplementedError('3 or 4 channel RAW images currently not supported')
            cdef np.npy_intp shape[2]
            shape[0] = <np.npy_intp> self.p.imgdata.sizes.raw_height
            shape[1] = <np.npy_intp> self.p.imgdata.sizes.raw_width
            cdef np.ndarray ndarr
            ndarr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, raw)
            # ndarr must hold a reference to this object,
            # otherwise the underlying data gets lost when the RawPy instance gets out of scope
            # (which would trigger __dealloc__)
            ndarr.base = <PyObject*> self
            # Python doesn't know about above assignment as it's in C-level 
            Py_INCREF(self)
            return ndarr
    
    property raw_image_visible:
        """
        Like raw_image but without margin.
        
        :rtype: ndarray of shape (hv,wv)
        """
        def __get__(self):
            raw_image = self.raw_image
            if raw_image is None:
                return None
            s = self.sizes
            return raw_image[s.top_margin:s.top_margin+s.height,
                             s.left_margin:s.left_margin+s.width]
            
    cpdef ushort raw_value(self, int row, int column):
        """
        Return RAW value at given position relative to the full RAW image.
        Only usable for flat RAW images (see :attr:`~rawpy.RawPy.raw_type` property).
        """
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        if raw == NULL:
            raise RuntimeError('RAW image is not flat')
        cdef ushort raw_width = self.p.imgdata.sizes.raw_width
        return raw[row*raw_width + column]
            
    cpdef ushort raw_value_visible(self, int row, int column):
        """
        Return RAW value at given position relative to visible area of image.
        Only usable for flat RAW images (see raw_type property).        
        """
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        if raw == NULL:
            raise RuntimeError('RAW image is not flat')
        cdef ushort top_margin = self.p.imgdata.sizes.top_margin
        cdef ushort left_margin = self.p.imgdata.sizes.left_margin
        cdef ushort raw_width = self.p.imgdata.sizes.raw_width
        return raw[(row+top_margin)*raw_width + column + left_margin]
        
    property sizes:
        """
        Return a :class:`rawpy.ImageSizes` instance with size information of
        the RAW image and postprocessed image.        
        """
        def __get__(self):
            cdef libraw_image_sizes_t* s = &self.p.imgdata.sizes
            return ImageSizes(raw_height=s.raw_height, raw_width=s.raw_width,
                              height=s.height, width=s.width,
                              top_margin=s.top_margin, left_margin=s.left_margin,
                              iheight=s.iheight, iwidth=s.iwidth,
                              pixel_aspect=s.pixel_aspect, flip=s.flip)
    
    property num_colors:
        """
        Number of colors.
        Note that e.g. for RGBG this can be 3 or 4, depending on the camera model,
        as some use two different greens. 
        """
        def __get__(self):
            return self.p.imgdata.idata.colors
    
    property color_desc:
        """
        String description of colors numbered from 0 to 3 (RGBG,RGBE,GMCY, or GBTG).
        Note that same letters may not refer strictly to the same color.
        There are cameras with two different greens for example.
        """
        def __get__(self):
            return self.p.imgdata.idata.cdesc
    
    cpdef int raw_color(self, int row, int column):
        """
        Return color index for the given coordinates relative to the full RAW size.
        Only usable for flat RAW images (see raw_type property).
        """
        if self.p.imgdata.rawdata.raw_image == NULL:
            raise RuntimeError('RAW image is not flat')
        return self.p.COLOR(row, column)
    
    property raw_colors:
        """
        An array of color indices for each pixel in the RAW image.
        Equivalent to calling raw_color(y,x) for each pixel.
        Only usable for flat RAW images (see raw_type property).
        
        :rtype: ndarray of shape (h,w)
        """
        def __get__(self):
            if self.p.imgdata.rawdata.raw_image == NULL:
                raise RuntimeError('RAW image is not flat')
            cdef np.ndarray pattern = self.raw_pattern
            cdef int n = pattern.shape[0]
            cdef int height = self.p.imgdata.sizes.raw_height
            cdef int width = self.p.imgdata.sizes.raw_width
            return np.tile(pattern, (height/n, width/n))
    
    property raw_colors_visible:
        """
        Like raw_colors but without margin.
        
        :rtype: ndarray of shape (hv,wv)
        """
        def __get__(self):
            s = self.sizes
            return self.raw_colors[s.top_margin:s.top_margin+s.height,
                                   s.left_margin:s.left_margin+s.width]
    
    property raw_pattern:
        """
        The smallest possible Bayer pattern of this image.
        
        :rtype: ndarray, or None if not a flat RAW image
        """
        def __get__(self):
            if self.p.imgdata.rawdata.raw_image == NULL:
                return None
            cdef np.ndarray pattern
            cdef int n
            if self.p.imgdata.idata.filters < 1000:
                if self.p.imgdata.idata.filters == 0:
                    # black and white
                    n = 1
                if self.p.imgdata.idata.filters == 1:
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
                    pattern[y,x] = self.p.COLOR(y, x)
            if n == 4:
                if np.all(pattern[:2,:2] == pattern[:2,2:]) and \
                   np.all(pattern[:2,:2] == pattern[2:,2:]) and \
                   np.all(pattern[:2,:2] == pattern[2:,:2]):
                    pattern = pattern[:2,:2]
            return pattern
       
    property camera_whitebalance:
        """
        White balance coefficients (as shot). Either read from file or calculated.
        
        :rtype: list of length 4
        """
        def __get__(self):
            return [self.p.imgdata.rawdata.color.cam_mul[0],
                    self.p.imgdata.rawdata.color.cam_mul[1],
                    self.p.imgdata.rawdata.color.cam_mul[2],
                    self.p.imgdata.rawdata.color.cam_mul[3]]
        
    property daylight_whitebalance:
        """
        White balance coefficients for daylight (daylight balance). 
        Either read from file, or calculated on the basis of file data, 
        or taken from hardcoded constants.
        
        :rtype: list of length 4
        """
        def __get__(self):
            return [self.p.imgdata.rawdata.color.pre_mul[0],
                    self.p.imgdata.rawdata.color.pre_mul[1],
                    self.p.imgdata.rawdata.color.pre_mul[2],
                    self.p.imgdata.rawdata.color.pre_mul[3]]
            
    property black_level_per_channel:
        """
        Per-channel black level correction.
        NOTE: This equals black + cblack[N] in LibRaw.
        
        :rtype: list of length 4
        """
        def __get__(self):
            cdef unsigned black = self.p.imgdata.rawdata.color.black
            return [black + self.p.imgdata.rawdata.color.cblack[0],
                    black + self.p.imgdata.rawdata.color.cblack[1],
                    black + self.p.imgdata.rawdata.color.cblack[2],
                    black + self.p.imgdata.rawdata.color.cblack[3]]
            
    property color_matrix:
        """
        Color matrix, read from file for some cameras, calculated for others. 
        
        :rtype: ndarray of shape (3,4)
        """
        def __get__(self):
            cdef np.ndarray matrix = np.empty((3, 4), dtype=np.float32)
            for i in range(3):
                for j in range(4):
                    matrix[i,j] = self.p.imgdata.rawdata.color.cmatrix[i][j]
            return matrix
        
    property rgb_xyz_matrix:
        """
        Camera RGB - XYZ conversion matrix.
        This matrix is constant (different for different models).
        Last row is zero for RGB cameras and non-zero for different color models (CMYG and so on).
        
        :rtype: ndarray of shape (4,3)
        """
        def __get__(self):
            cdef np.ndarray matrix = np.empty((4, 3), dtype=np.float32)
            for i in range(4):
                for j in range(3):
                    matrix[i,j] = self.p.imgdata.rawdata.color.cam_xyz[i][j]
            return matrix
    
    property tone_curve:
        """
        Camera tone curve, read from file for Nikon, Sony and some other cameras.
        
        :rtype: ndarray of length 65536
        """
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> 65536
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_USHORT,
                                                &self.p.imgdata.rawdata.color.curve)

    def dcraw_process(self, params=None, **kw):
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
        if libraw_version < (0,15):
            if self.needs_reopening:
                warnings.warn('Repeated postprocessing with libraw<0.15 may require reopening/unpacking')
            self.needs_reopening = True
        if params and kw:
            raise ValueError('If params is given, then no additional keywords are allowed')
        if params is None:
            params = Params(**kw)
        self.apply_params(params)
        self.handle_error(self.p.dcraw_process())
        
    def dcraw_make_mem_image(self):
        """
        Return the postprocessed image (see :meth:`~rawpy.RawPy.dcraw_process`) as numpy array.
        
        .. NOTE:: This is a low-level method, consider using :meth:`~rawpy.RawPy.postprocess` instead.
        
        :rtype: ndarray of shape (h,w,c)
        """
        cdef int errcode = 0
        cdef libraw_processed_image_t* img = self.p.dcraw_make_mem_image(&errcode)
        self.handle_error(errcode)
        if img.type != LIBRAW_IMAGE_BITMAP:
            raise NotImplementedError
        wrapped = processed_image_wrapper()
        wrapped.set_data(self, img)
        ndarr = wrapped.__array__()
        return ndarr
    
    def postprocess(self, params=None, **kw):
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
        p.use_camera_wb = params.use_camera_wb
        p.use_auto_wb = params.use_auto_wb
        if params.user_mul:
            for i in range(4):
                p.user_mul[i] = params.user_mul[i]
        p.output_color = params.output_color
        p.output_bps = params.output_bps
        p.user_flip = params.user_flip
        p.user_black = params.user_black
        p.user_sat = params.user_sat
        p.no_auto_bright = params.no_auto_bright
        p.auto_bright_thr = params.auto_bright_thr
        p.adjust_maximum_thr = params.adjust_maximum_thr
        p.bright = params.bright
        p.exp_correc = params.exp_correc
        p.exp_shift = params.exp_shift
        p.exp_preser = params.exp_preser
        if params.bad_pixels:
            p.bad_pixels = params.bad_pixels
        else:
            p.bad_pixels = NULL
        p.gamm[0] = params.gamm[0]
        p.gamm[1] = params.gamm[1]
    
    cdef handle_error(self, int code):
        if code > 0:
            raise OSError((code, os.strerror(code))) 
        elif code < 0:
            errstr = self.p.strerror(code)
            if code < -10000: # see macro LIBRAW_FATAL_ERROR in libraw_const.h
                raise LibRawFatalError(errstr)
            else:
                print(repr(LibRawNonFatalError(errstr)), file=sys.stderr)

class DemosaicAlgorithm(Enum):
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
    
class ColorSpace(Enum):
    raw=0
    sRGB=1
    Adobe=2
    Wide=3
    ProPhoto=4
    XYZ=5

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
    def __init__(self, demosaic_algorithm=None,
                 use_camera_wb=False, use_auto_wb=False, user_wb=None,
                 output_color=ColorSpace.sRGB, output_bps=8, 
                 user_flip=None, user_black=None, user_sat=None,
                 no_auto_bright=False, auto_bright_thr=None, adjust_maximum_thr=0.75,
                 bright=None,
                 exp_shift=None, exp_preserve_highlights=0.0,
                 gamma=None,
                 bad_pixels_path=None):
        """

        If use_camera_wb and use_auto_wb are False and user_wb is None, then
        daylight white balance correction is used.
        If both use_camera_wb and use_auto_wb are True, then use_auto_wb has priority.
        
        :param rawpy.DemosaicAlgorithm demosaic_algorithm: default is AHD
        :param bool use_camera_wb: whether to use the as-shot white balance values
        :param bool use_auto_wb: whether to try automatically calculating the white balance 
        :param list user_wb: list of length 4 with white balance multipliers for each color 
        :param rawpy.ColorSpace output_color: output color space
        :param int output_bps: 8 or 16
        :param int user_flip: 0=none, 3=180, 5=90CCW, 6=90CW,
                              default is to use image orientation from the RAW image if available
        :param int user_black: custom black level
        :param int user_sat: saturation adjustment
        :param bool no_auto_bright: whether to disable automatic increase of brightness
        :param float auto_bright_thr: ratio of clipped pixels when automatic brighness increase is used
                                      (see `no_auto_bright`). Default is 0.01 (1%).
        :param float adjust_maximum_thr: see libraw docs
        :param float bright: brightness (default 1.0)
        :param float exp_shift: exposure shift in linear scale.
                          Usable range from 0.25 (2-stop darken) to 8.0 (3-stop lighter).
        :param float exp_preserve_highlights: preserve highlights when lightening the image with `exp_shift`.
                          From 0.0 to 1.0 (full preservation).
        :param tuple gamma: pair (power,slope), default is (2.222, 4.5) for rec. BT.709
        :param str bad_pixels_path: path to dcraw bad pixels file. Each bad pixel will be corrected using
                                    the mean of the neighbor pixels. See the :mod:`rawpy.enhance` module
                                    for alternative repair algorithms, e.g. using the median.
        """

        if demosaic_algorithm:
            demosaic_algorithm.checkSupported()
            self.user_qual = demosaic_algorithm.value
        else:
            self.user_qual = -1
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
        self.user_sat = user_sat if user_sat is not None else -1
        self.no_auto_bright = no_auto_bright
        if auto_bright_thr is not None:
            min_version = (0,16,1)
            if libraw_version < min_version:
                # see https://github.com/LibRaw/LibRaw/commit/ea70421a518ba5a039fcc1dc1045b428159fb032
                raise NotSupportedError('Parameter auto_bright_thr', min_version)
            self.auto_bright_thr = auto_bright_thr
        else:
            self.auto_bright_thr = LIBRAW_DEFAULT_AUTO_BRIGHTNESS_THRESHOLD
        self.adjust_maximum_thr = adjust_maximum_thr
        self.bright = bright if bright is not None else 1.0
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
        self.bad_pixels = bad_pixels_path
    
class LibRawFatalError(Exception):
    pass

class LibRawNonFatalError(Exception):
    pass                
    
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
        ndarr.base = <PyObject*> self
        # Python doesn't know about above assignment as it's in C-level 
        Py_INCREF(self)
        return ndarr

    def __dealloc__(self):
        self.raw.p.dcraw_clear_mem(self.processed_image)        
    
cdef char* _chars(s):
    if isinstance(s, unicode):
        # convert unicode to chars
        s = (<unicode>s).encode('UTF-8')
    return s

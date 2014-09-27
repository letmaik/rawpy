# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
# cython: embedsignature=True

from __future__ import print_function

from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np
np.import_array()

import os
import sys
from enum import Enum

cdef extern from "libraw.h":
    ctypedef unsigned short ushort
    
    cdef enum LibRaw_image_formats:
        LIBRAW_IMAGE_JPEG
        LIBRAW_IMAGE_BITMAP
    
    ctypedef struct libraw_image_sizes_t:
        ushort raw_height, raw_width
        ushort height, width
        ushort top_margin, left_margin
        ushort iheight, iwidth

    ctypedef struct libraw_rawdata_t:
        ushort *raw_image
        
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

    ctypedef struct libraw_data_t:
        ushort                      (*image)[4]
        libraw_image_sizes_t        sizes
#         libraw_iparams_t            idata
        libraw_output_params_t        params
#         unsigned int                progress_flags
#         unsigned int                process_warnings
#         libraw_colordata_t          color
#         libraw_imgother_t           other
#         libraw_thumbnail_t          thumbnail
        libraw_rawdata_t            rawdata
#         void                *parent_class

    ctypedef struct libraw_processed_image_t:
        #enum LibRaw_image_formats type
        ushort height, width, colors, bits
        unsigned int  data_size 
        unsigned char data[1] # this is the image data, no idea why [1]

    cdef cppclass LibRaw:
        libraw_data_t imgdata
        LibRaw()
        int open_file(const char *fname)
        int unpack()
        int COLOR(int row, int col)
        int subtract_black()
        int raw2image()
        int raw2image_ex(int do_subtract_black)
        void raw2image_start()
        int dcraw_process()
        libraw_processed_image_t* dcraw_make_mem_image(int *errcode)
        void dcraw_clear_mem(libraw_processed_image_t* img)
        void free_image()
        const char* strerror(int p)
        
        # debugging:
        int                         dcraw_ppm_tiff_writer(const char *filename)
   
cdef class RawPy:
    cdef LibRaw* p
    cdef np.ndarray _raw_image
        
    def __cinit__(self):
        self.p = new LibRaw()
        
    def __dealloc__(self):
        del self.p
    
    def open_file(self, path):
        # TODO check error code and turn into exception
        self.handleError(self.p.open_file(_chars(path)))
        
    def unpack(self):
        self.handleError(self.p.unpack())
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.p.imgdata.sizes.raw_height
        shape[1] = <np.npy_intp> self.p.imgdata.sizes.raw_width
        self._raw_image = np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, raw)
        
    @property
    def raw_image(self):
        """Bayer-pattern RAW image, one channel."""
        return self._raw_image
    
    cpdef ushort rawvalue(self, int row, int column):
        """
        Return RAW value at given position relative to visible area of image
        (see visible_size_raw()).        
        """
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        cdef ushort top_margin = self.p.imgdata.sizes.top_margin
        cdef ushort left_margin = self.p.imgdata.sizes.left_margin
        cdef ushort raw_width = self.p.imgdata.sizes.raw_width
        return raw[(row+top_margin)*raw_width + column + left_margin]
    
    @property
    def visible_size_raw(self):
        return self.p.imgdata.sizes.height, self.p.imgdata.sizes.width
    
    cpdef int rawcolor(self, int row, int column):
        return self.p.COLOR(row, column)
    
    def raw2composite(self):
        """
        Creates a RAW composite image with RGBG channels, accessible via .image property.
        """
        self.handleError(self.p.raw2image())
    
    @property
    def composite_image(self):
        """
        RAW composite image with RGBG channels. 
        Note that this image contains duplicated pixels such that it
        matches the visible RAW image size."""
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.p.imgdata.sizes.iheight
        shape[1] = <np.npy_intp> self.p.imgdata.sizes.iwidth
        return [np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[0]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[1]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[2]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[3])]
        
    def subtract_black(self):
        self.handleError(self.p.subtract_black())
        
    def dcraw_process(self, params=None):
        self.applyParams(params)
        self.handleError(self.p.dcraw_process())
        
    def dcraw_make_mem_image(self):
        # TODO how can it be that unsigned char represent 16 bits?
        cdef int errcode = 0
        cdef libraw_processed_image_t* img = self.p.dcraw_make_mem_image(&errcode)
        self.handleError(errcode)
        #assert img.type == LIBRAW_IMAGE_BITMAP
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> img.height
        shape[1] = <np.npy_intp> img.width
        shape[2] = <np.npy_intp> img.colors
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_UINT8 if img.bits == 8 else np.NPY_UINT16, img.data)
        # FIXME call dcraw_clear_mem
        return arr
    
    def dcraw_ppm_tiff_writer(self, const char *filename):
        self.handleError(self.p.dcraw_ppm_tiff_writer(filename))        
    
    cdef applyParams(self, params):
        if params is None:
            return
        cdef libraw_output_params_t* p = &self.p.imgdata.params
        if params.user_qual is not None:
            p.user_qual = params.user_qual
            assert self.p.imgdata.params.user_qual == params.user_qual
    
    cdef handleError(self, int code):
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

class Params(object):
    def __init__(self, demosaic_algorithm=None):
        self.user_qual = demosaic_algorithm        
    
class LibRawFatalError(Exception):
    pass

class LibRawNonFatalError(Exception):
    pass                
    
cdef char* _chars(s):
    if isinstance(s, unicode):
        # convert unicode to chars
        s = (<unicode>s).encode('UTF-8')
    return s    
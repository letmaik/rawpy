# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
# cython: embedsignature=True

from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np
np.import_array()

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

    ctypedef struct libraw_data_t:
        ushort                      (*image)[4]
        libraw_image_sizes_t        sizes
#         libraw_iparams_t            idata
#         libraw_output_params_t        params
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
        int subtract_black()
        int raw2image()
        int raw2image_ex(int do_subtract_black)
        void raw2image_start()
        int dcraw_process()
        libraw_processed_image_t* dcraw_make_mem_image(int *errcode=NULL)
        void dcraw_clear_mem(libraw_processed_image_t* img)
        void free_image()
        
def enum(**enums):
    return type('Enum', (), enums)

def enumKey(enu, val):
    # cython doesn't like tuple unpacking in lambdas ("Expected ')', found ','")
    #return filter(lambda (k,v): v == val, enu.__dict__.items())[0][0]
    return filter(lambda item: item[1] == val, enu.__dict__.items())[0][0]
   
cdef class RawPy:
    cdef LibRaw* p
        
    def __cinit__(self):
        self.p = new LibRaw()
        
    def __dealloc__(self):
        del self.p
    
    def open_file(self, path):
        # TODO check error code and turn into exception
        self.p.open_file(_chars(path))
        
    def unpack(self):
        self.p.unpack()
        
    @property
    def rawdata(self):
        """Bayer-pattern RAW image, one channel."""
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.p.imgdata.sizes.raw_height
        shape[1] = <np.npy_intp> self.p.imgdata.sizes.raw_width
        return np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, raw)
    
    def raw2image(self):
        self.p.raw2image()
    
    @property
    def image(self):
        """RAW composite image with channels RGBR."""
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.p.imgdata.sizes.iheight
        shape[1] = <np.npy_intp> self.p.imgdata.sizes.iwidth
        return [np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[0]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[1]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[2]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[3])]
        
    def subtract_black(self):
        self.p.subtract_black()
        
    def dcraw_process(self):
        self.p.dcraw_process()
        
    def dcraw_make_mem_image(self):
        # TODO how can it be that unsigned char represent 16 bits?
        cdef libraw_processed_image_t* img = self.p.dcraw_make_mem_image()
        #assert img.type == LIBRAW_IMAGE_BITMAP
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> img.height
        shape[1] = <np.npy_intp> img.width
        shape[2] = <np.npy_intp> img.colors
        # FIXME memory leak, numpy doesn't own img.data! we have to free it later using dcraw_clear_mem()!
        #       -> write wrapper class with __array__
        return np.PyArray_SimpleNewFromData(3, shape, np.NPY_UINT8 if img.bits == 8 else np.NPY_UINT16, img.data)
        
cdef char* _chars(s):
    if isinstance(s, unicode):
        # convert unicode to chars
        s = (<unicode>s).encode('UTF-8')
    return s

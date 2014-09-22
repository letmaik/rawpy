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
        unsigned raw_pitch

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
        int COLOR(int row, int col)
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
    cdef np.ndarray _raw_image
        
    def __cinit__(self):
        self.p = new LibRaw()
        
    def __dealloc__(self):
        del self.p
    
    def open_file(self, path):
        # TODO check error code and turn into exception
        self.p.open_file(_chars(path))
        
    def unpack(self):
        self.p.unpack()
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
        cdef ushort pitch = self.p.imgdata.sizes.raw_pitch/2
        return raw[(row+top_margin)*pitch + column + left_margin]
    
    @property
    def visible_size_raw(self):
        return self.p.imgdata.sizes.height, self.p.imgdata.sizes.width
    
    cpdef int rawcolor(self, int row, int column):
        return self.p.COLOR(row, column)
    
    def bench(self):
        # just a small benchmark..
        cdef int row, col, cnt
        h,w = self.visible_size_raw
        
        for row in range(h):
            for col in range(w):
                if self.rawcolor(row,col) == 3 and self.rawvalue(row,col) > 100:
                    cnt += 1
        return cnt
    
    def raw2composite(self):
        """
        Creates a RAW composite image with RGBG channels, accessible via .image property.
        """
        self.p.raw2image()
    
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

# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8
# cython: embedsignature=True

from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "libraw.h":
    ctypedef unsigned short ushort
    
    struct libraw_image_sizes_t:
        ushort raw_height, raw_width
        ushort height, width 
        ushort top_margin, left_margin

    struct libraw_rawdata_t:
        ushort *raw_image

    struct libraw_data_t:
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

    cdef cppclass LibRaw:
        libraw_data_t imgdata
        LibRaw()
        int open_file(const char *fname)
        int unpack()
        int subtract_black()
        int raw2image()
        int raw2image_ex(int do_subtract_black)
        void raw2image_start()
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
        self.p.open_file(_chars(path))
        
    def unpack(self):
        self.p.unpack()
        
    @property
    def rawdata(self):
        cdef ushort* raw = self.p.imgdata.rawdata.raw_image
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.p.imgdata.sizes.raw_height
        shape[1] = <np.npy_intp> self.p.imgdata.sizes.raw_width
        return np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, raw)
    
    @property
    def image(self):
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.p.imgdata.sizes.height
        shape[1] = <np.npy_intp> self.p.imgdata.sizes.width
        return [np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[0]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[1]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[2]),
                np.PyArray_SimpleNewFromData(2, shape, np.NPY_USHORT, self.p.imgdata.image[3])]
        
    def subtract_black(self):
        self.p.subtract_black()
        
    def raw2image(self):
        self.p.raw2image()
        
cdef char* _chars(s):
    if isinstance(s, unicode):
        # convert unicode to chars
        s = (<unicode>s).encode('UTF-8')
    return s

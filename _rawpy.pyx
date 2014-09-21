# cython: c_string_type=unicode, c_string_encoding=utf8
# cython: embedsignature=True

from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
from collections import namedtuple
cimport numpy as np
np.import_array()

# We cannot use Cython's C++ support here, as libraw.h exposes functions
# in an 'extern "C"' block, which is not supported yet. Therefore, libraw's
# C interface is used.

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef extern from "libraw.h":
    struct libraw_data_t:
        pass
            
    libraw_data_t *libraw_init(unsigned int flags)
    
def enum(**enums):
    return type('Enum', (), enums)

def enumKey(enu, val):
    # cython doesn't like tuple unpacking in lambdas ("Expected ')', found ','")
    #return filter(lambda (k,v): v == val, enu.__dict__.items())[0][0]
    return filter(lambda item: item[1] == val, enu.__dict__.items())[0][0]

        
cdef char* _chars(s):
    if isinstance(s, unicode):
        # convert unicode to chars
        s = (<unicode>s).encode('UTF-8')
    return s

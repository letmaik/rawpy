from __future__ import division, print_function, absolute_import

import os
import numpy as np
from numpy.testing.utils import assert_array_equal

import rawpy

rawTestPath = os.path.join(os.path.dirname(__file__), 'iss030e122639.NEF')

def testFileOpen():
    raw = rawpy.imread(rawTestPath)
    assert_array_equal(raw.raw_image.shape, [2844, 4288])
    
    raw.dcraw_process()
    rgb = raw.dcraw_make_mem_image()
    assert_array_equal(rgb.shape, [2844, 4284, 3])
        
if __name__ == '__main__':
    testFileOpen()
    
from __future__ import division, print_function

import os
import numpy as np
from numpy.testing.utils import assert_array_equal

import rawpy

rawTestPath = os.path.join(os.path.dirname(__file__), 'iss030e122639.NEF')

def testFileOpen():
    raw = rawpy.imread(rawTestPath)
    assert_array_equal(raw.rawdata.shape, [2844, 4288])
    print(np.min(raw.rawdata), np.max(raw.rawdata))
    
    # composite RAW image with RGBR channels
    raw.raw2image()
    for img_channel in raw.image:
        print(img_channel.shape)
        print(np.min(img_channel), np.max(img_channel), np.mean(img_channel))
    
    # run bayer interpolation and get the RGB image
    raw.dcraw_process()
    rgb = raw.dcraw_make_mem_image()
    print(rgb.dtype, rgb.shape)
    for i in range(rgb.shape[2]):
        print(np.min(rgb[:,:,i]), np.max(rgb[:,:,i]), np.mean(rgb[:,:,i]))
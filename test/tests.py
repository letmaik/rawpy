from __future__ import division, print_function

import os
import numpy as np
from numpy.testing.utils import assert_array_equal

import rawpy

rawTestPath = os.path.join(os.path.dirname(__file__), 'iss030e122639.NEF')

def testFileOpen():
    raw = rawpy.imread(rawTestPath)
    assert_array_equal(raw.raw_image.shape, [2844, 4288])
    print(np.min(raw.raw_image), np.max(raw.raw_image))

    for r in range(1000,1010):
        for c in range(1000,1010):
            print(r,',',c,':',raw.rawvalue(r,c), raw.rawcolor(r,c))
    
    print(raw.bench())
    return
    
    # composite RAW image with RGBR channels
    raw.raw2composite()
    for img_channel in raw.composite_image:
        print(img_channel.shape)
        print(np.min(img_channel), np.max(img_channel), np.mean(img_channel))
    
    # run bayer interpolation and get the RGB image
    raw.dcraw_process()
    rgb = raw.dcraw_make_mem_image()
    print(rgb.dtype, rgb.shape)
    for i in range(rgb.shape[2]):
        print(np.min(rgb[:,:,i]), np.max(rgb[:,:,i]), np.mean(rgb[:,:,i]))
        
if __name__ == '__main__':
    testFileOpen()
    
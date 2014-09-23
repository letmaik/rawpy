from __future__ import division, print_function

import time
import os
import numpy as np
from numpy.testing.utils import assert_array_equal
from PIL import Image

import rawpy

rawTestPath = os.path.join(os.path.dirname(__file__), 'iss030e122639.NEF')

def testFileOpen():
    raw = rawpy.imread(rawTestPath)
    assert_array_equal(raw.raw_image.shape, [2844, 4288])
    print(np.min(raw.raw_image), np.max(raw.raw_image))

    for r in range(1000,1010):
        for c in range(1000,1010):
            print(r,',',c,':',raw.rawvalue(r,c), raw.rawcolor(r,c))
    
    # FIXME all files are identical, what's going on??
    for alg in rawpy.DemosaicAlgorithm:
        t0 = time.time()
        params = rawpy.Params(demosaic_algorithm=alg.value)
        raw.dcraw_process(params)
        rgb = raw.dcraw_make_mem_image()
        print(alg.name, 'demosaic:', time.time()-t0, 's')
        
        print(rgb.dtype, rgb.shape)
        Image.fromarray(rgb).save('test_demosaic_' + alg.name + '.png')


def _testComposite():
    raw = rawpy.imread(rawTestPath)
    # composite RAW image with RGBR channels
    raw.raw2composite()
    for img_channel in raw.composite_image:
        print(img_channel.shape)
        print(np.min(img_channel), np.max(img_channel), np.mean(img_channel))    
        
if __name__ == '__main__':
    testFileOpen()
    
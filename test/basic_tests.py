from __future__ import division, print_function, absolute_import

import os
import numpy as np
from numpy.testing.utils import assert_array_equal

from PIL import Image
import rawpy

rawTestPath = os.path.join(os.path.dirname(__file__), 'iss030e122639.NEF')

def testFileOpenAndPostProcess():
    raw = rawpy.imread(rawTestPath)
    assert_array_equal(raw.raw_image.shape, [2844, 4288])
    
    rgb = raw.postprocess(no_auto_bright=True)
    assert_array_equal(rgb.shape, [2844, 4284, 3])
    print(np.min(rgb, axis=(0,1)), np.max(rgb, axis=(0,1)), np.sum(rgb==255, axis=(0,1)))
    Image.fromarray(rgb).save('test_no_auto_bright.png')
    
    # FIXME auto_bright_thr has no effect
    rgb = raw.postprocess(no_auto_bright=False, auto_bright_thr=0.0)
    print(np.min(rgb, axis=(0,1)), np.max(rgb, axis=(0,1)), np.sum(rgb==255, axis=(0,1)))
    Image.fromarray(rgb).save('test_auto_bright_0.0.png')
    
    rgb = raw.postprocess(no_auto_bright=False, auto_bright_thr=0.5)
    print(np.min(rgb, axis=(0,1)), np.max(rgb, axis=(0,1)), np.sum(rgb==255, axis=(0,1)))
    Image.fromarray(rgb).save('test_auto_bright_0.01.png')
        
if __name__ == '__main__':
    testFileOpenAndPostProcess()
    
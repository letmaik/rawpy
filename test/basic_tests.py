from __future__ import division, print_function, absolute_import

import os
import numpy as np
from numpy.testing.utils import assert_array_equal

import rawpy
import imageio

rawTestPath = os.path.join(os.path.dirname(__file__), 'iss030e122639.NEF')

def testVersion():
    print('using libraw', rawpy.libraw_version)  

def testFileOpenAndPostProcess():
    raw = rawpy.imread(rawTestPath)
    assert_array_equal(raw.raw_image.shape, [2844, 4288])   
    
    rgb = raw.postprocess(no_auto_bright=True, user_wb=raw.daylight_whitebalance)
    assert_array_equal(rgb.shape, [2844, 4284, 3])
    print_stats(rgb)
    save('test_8daylight.tiff', rgb)

    print('daylight white balance multipliers:', raw.daylight_whitebalance)
    
    rgb = raw.postprocess(no_auto_bright=True, user_wb=raw.daylight_whitebalance)
    print_stats(rgb)
    save('test_8daylight2.tiff', rgb)
 
    rgb = raw.postprocess(no_auto_bright=True, user_wb=raw.daylight_whitebalance,
                          output_bps=16)
    print_stats(rgb)
    save('test_16daylight.tiff', rgb)
     
    # linear images are more useful for science (=no gamma correction)
    # see http://www.mit.edu/~kimo/blog/linear.html
    rgb = raw.postprocess(no_auto_bright=True, user_wb=raw.daylight_whitebalance,
                          gamma=(1,1), output_bps=16)
    print_stats(rgb)
    save('test_16daylight_linear.tiff', rgb)


def save(path, im):
    # both imageio and skimage currently save uint16 images with 180deg rotation
    # as they both use freeimage and this has some weird internal formats
    # see https://github.com/scikit-image/scikit-image/issues/1101
    # and https://github.com/imageio/imageio/issues/3
    from distutils.version import StrictVersion
    if im.dtype == np.uint16 and StrictVersion(imageio.__version__) <= StrictVersion('0.5.1'):
        im = im[::-1,::-1]
    imageio.imsave(path, im)

def print_stats(rgb):
    print(rgb.dtype, 
          np.min(rgb, axis=(0,1)), np.max(rgb, axis=(0,1)), # range for each channel
          [len(np.unique(rgb[:,:,0])), len(np.unique(rgb[:,:,1])), len(np.unique(rgb[:,:,2]))], # unique values
          np.sum(rgb==np.iinfo(rgb.dtype).max, axis=(0,1))) # number of saturated pixels
        
if __name__ == '__main__':   
    testFileOpenAndPostProcess()
    
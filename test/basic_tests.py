from __future__ import division, print_function, absolute_import

import os
from pprint import pprint
import numpy as np
import numpy.ma as ma
from numpy.testing.utils import assert_array_equal, assert_equal

import rawpy
import rawpy.enhance
import imageio
from rawpy.enhance import _repair_bad_pixels_bayer2x2,\
    _repair_bad_pixels_generic

rawTestPath = os.path.join(os.path.dirname(__file__), 'iss030e122639.NEF')
badPixelsTestPath = os.path.join(os.path.dirname(__file__), 'bad_pixels.gz')

def testVersion():
    print('using libraw', rawpy.libraw_version)
    pprint(rawpy.flags)
    for d in rawpy.DemosaicAlgorithm:
        print(d.name, 'NOT' if d.isSupported is False 
                      else 'possibly' if d.isSupported is None else '', 'supported')

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

def testBadPixelRepair():
    def getColorNeighbors(raw, y, x):
        # 5x5 area around coordinate masked by color of coordinate
        raw_colors = raw.raw_colors_visible
        raw_color = raw_colors[y, x]
        masked = ma.masked_array(raw.raw_image_visible, raw_colors!=raw_color)
        return masked[y-2:y+3,x-2:x+3].copy()
    
    bad_pixels = np.loadtxt(badPixelsTestPath, int)
    i = 60
    y, x = bad_pixels[i,0], bad_pixels[i,1]
    
    for useOpenCV in [False,True]:
        if useOpenCV:
            if rawpy.enhance.cv2 is None:
                print('OpenCV not available, skipping subtest')
                continue
            print('testing with OpenCV')
        else:
            print('testing without OpenCV')
            oldCv = rawpy.enhance.cv2
            rawpy.enhance.cv2 = None
            
        for repair in [_repair_bad_pixels_generic, _repair_bad_pixels_bayer2x2]:
            print('testing ' + repair.__name__)
            raw = rawpy.imread(rawTestPath)
            
            before = getColorNeighbors(raw, y, x)
            repair(raw, bad_pixels, method='median')
            after = getColorNeighbors(raw, y, x)
        
            print(before)
            print(after)
        
            # check that the repaired value is the median of the 5x5 neighbors
            assert_equal(int(ma.median(before)), raw.raw_image_visible[y,x],
                         'median wrong for ' + repair.__name__)
        if not useOpenCV:
            rawpy.enhance.cv2 = oldCv

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
    testVersion()
    #testFileOpenAndPostProcess()
    testBadPixelRepair()
    
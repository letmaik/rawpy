from __future__ import division, print_function, absolute_import

import os
import shutil
import pytest
import numpy as np
import numpy.ma as ma
from pprint import pprint
from numpy.testing import assert_array_equal, assert_equal

import rawpy
import rawpy.enhance
import imageio.v3 as iio
from rawpy.enhance import _repair_bad_pixels_bayer2x2,\
    _repair_bad_pixels_generic, find_bad_pixels

thisDir = os.path.dirname(__file__)

# Nikon D3S
rawTestPath = os.path.join(thisDir, 'iss030e122639.NEF')
badPixelsTestPath = os.path.join(thisDir, 'bad_pixels.gz')

# Nikon D4
raw2TestPath = os.path.join(thisDir, 'iss042e297200.NEF')

# Canon EOS 5D Mark 2
raw3TestPath = os.path.join(thisDir, 'RAW_CANON_5DMARK2_PREPROD.CR2')

# Sigma SD9 (Foveon)
raw4TestPath = os.path.join(thisDir, 'RAW_SIGMA_SD9_SRGB.X3F')

# Canon 40D in sRAW format with 4 channels
raw5TestPath = os.path.join(thisDir, 'RAW_CANON_40D_SRAW_V103.CR2')

# Kodak DC50 with special characters in filename
raw6TestPath = os.path.join(thisDir, 'RAW_KODAK_DC50_é.KDC')

# CR2 file with corrupted data
raw7TestPath = os.path.join(thisDir, 'M0054341_01_00005.cr2')

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
    iio.imwrite('test_8daylight.tiff', rgb)
 
    print('daylight white balance multipliers:', raw.daylight_whitebalance)
     
    rgb = raw.postprocess(no_auto_bright=True, user_wb=raw.daylight_whitebalance)
    print_stats(rgb)
    iio.imwrite('test_8daylight2.tiff', rgb)
 
    rgb = raw.postprocess(no_auto_bright=True, user_wb=raw.daylight_whitebalance,
                          output_bps=16)
    print_stats(rgb)
    iio.imwrite('test_16daylight.tiff', rgb)
     
    # linear images are more useful for science (=no gamma correction)
    # see http://www.mit.edu/~kimo/blog/linear.html
    rgb = raw.postprocess(no_auto_bright=True, user_wb=raw.daylight_whitebalance,
                          gamma=(1,1), output_bps=16)
    print_stats(rgb)
    iio.imwrite('test_16daylight_linear.tiff', rgb)

def testFoveonFileOpenAndPostProcess():
    raw = rawpy.imread(raw4TestPath)
    
    assert_array_equal(raw.raw_image.shape, [1531, 2304, 3])
    iio.imwrite('test_foveon_raw.tiff', raw.raw_image)
        
    rgb = raw.postprocess()
    assert_array_equal(rgb.shape, [1510, 2266, 3])
    print_stats(rgb)
    iio.imwrite('test_foveon.tiff', rgb)

def testSRawFileOpenAndPostProcess():
    raw = rawpy.imread(raw5TestPath)
    
    assert_array_equal(raw.raw_image.shape, [1296, 1944, 4])
    assert_equal(raw.raw_image[:,:,3].max(), 0)
    iio.imwrite('test_sraw_raw.tiff', raw.raw_image[:,:,:3])
        
    rgb = raw.postprocess()
    assert_array_equal(rgb.shape, [1296, 1944, 3])
    print_stats(rgb)
    iio.imwrite('test_sraw.tiff', rgb)

def testFileOpenWithNonAsciiCharacters():
    raw = rawpy.imread(raw6TestPath)

def testBufferOpen():
    with open(rawTestPath, 'rb') as rawfile:
        with rawpy.imread(rawfile) as raw:
            assert_array_equal(raw.raw_image.shape, [2844, 4288])
            rgb = raw.postprocess()
    print_stats(rgb)
    iio.imwrite('test_buffer.tiff', rgb)

def testContextManager():
    with rawpy.imread(rawTestPath) as raw:
        assert_array_equal(raw.raw_image.shape, [2844, 4288])

def testManualClose():
    raw = rawpy.imread(rawTestPath)
    assert_array_equal(raw.raw_image.shape, [2844, 4288])
    raw.close()
    
def testWindowsFileLockRelease():
    # see https://github.com/neothemachine/rawpy/issues/10
    # we make a copy of the raw file which we will later remove
    copyPath = rawTestPath + '-copy'
    shutil.copyfile(rawTestPath, copyPath)
    with rawpy.imread(copyPath) as raw:
        rgb = raw.postprocess()
    assert_array_equal(rgb.shape, [2844, 4284, 3])
    print_stats(rgb)
    # if the following does not throw an exception on Windows,
    # then the file is not locked anymore, which is how it should be
    os.remove(copyPath)
    
    # we test the same using .close() instead of a context manager
    shutil.copyfile(rawTestPath, copyPath)
    raw = rawpy.imread(copyPath)
    rgb = raw.postprocess()
    raw.close()
    os.remove(copyPath)
    assert_array_equal(rgb.shape, [2844, 4284, 3])

def testThumbExtractJPEG():
    with rawpy.imread(rawTestPath) as raw:
        thumb = raw.extract_thumb()
    assert thumb.format == rawpy.ThumbFormat.JPEG
    img = iio.imread(thumb.data)
    assert_array_equal(img.shape, [2832, 4256, 3])

def testThumbExtractBitmap():
    with rawpy.imread(raw4TestPath) as raw:
        thumb = raw.extract_thumb()
    assert thumb.format == rawpy.ThumbFormat.BITMAP
    assert_array_equal(thumb.data.shape, [378, 567, 3])

def testProperties():
    raw = rawpy.imread(rawTestPath)
    
    print('black_level_per_channel:', raw.black_level_per_channel)
    print('color_matrix:', raw.color_matrix)
    print('rgb_xyz_matrix:', raw.rgb_xyz_matrix)
    print('tone_curve:', raw.tone_curve)
    
    assert_array_equal(raw.black_level_per_channel, [0,0,0,0])
    
    # older versions have zeros at the end, was probably a bug
    if rawpy.libraw_version >= (0,16):
        assert_array_equal(raw.tone_curve, np.arange(65536))

def testBayerPattern():
    expected_desc = b'RGBG' # libraw hard-codes this and varies the color indices only
    
    for path in [rawTestPath, raw2TestPath]:
        raw = rawpy.imread(path)
        assert_equal(raw.color_desc, expected_desc)
        assert_array_equal(raw.raw_pattern, [[0,1],[3,2]])

    raw = rawpy.imread(raw3TestPath)
    assert_equal(raw.color_desc, expected_desc)
    assert_array_equal(raw.raw_pattern, [[3,2],[0,1]])

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

def testVisibleSize():
    for path in [rawTestPath, raw2TestPath]:
        print('testing', path)
        raw = rawpy.imread(path)
        s = raw.sizes
        print(s)
        h,w = raw.raw_image_visible.shape
        assert_equal(h, s.height)
        assert_equal(w, s.width)
        h,w = raw.raw_colors_visible.shape
        assert_equal(h, s.height)
        assert_equal(w, s.width)
        
def testCropSizeNikon():
    with rawpy.imread(rawTestPath) as raw:
        s = raw.sizes
        assert_equal(s.crop_left_margin, 0)
        assert_equal(s.crop_top_margin, 0)
        assert_equal(s.crop_width, 0)
        assert_equal(s.crop_height, 0)

def testCropSizeCanon():
    with rawpy.imread(raw3TestPath) as raw:
        s = raw.sizes
        assert_equal(s.crop_left_margin, 168)
        assert_equal(s.crop_top_margin, 56)
        assert_equal(s.crop_width, 5616)
        assert_equal(s.crop_height, 3744)

def testCropSizeSigma():
    with rawpy.imread(raw4TestPath) as raw:
        s = raw.sizes
        assert_equal(s.crop_left_margin, 0)
        assert_equal(s.crop_top_margin, 0)
        assert_equal(s.crop_width, 0)
        assert_equal(s.crop_height, 0)

def testCropSizeKodak():
    with rawpy.imread(raw6TestPath) as raw:
        s = raw.sizes
        assert_equal(s.crop_left_margin, 0)
        assert_equal(s.crop_top_margin, 0)
        assert_equal(s.crop_width, 0)
        assert_equal(s.crop_height, 0)

def testHalfSizeParameter():
    raw = rawpy.imread(rawTestPath)
    s = raw.sizes
    rgb = raw.postprocess(half_size=True)
    assert_equal(rgb.shape[0], s.height//2)
    assert_equal(rgb.shape[1], s.width//2)
    
def testHighlightModeParameter():
    raw = rawpy.imread(rawTestPath)
    raw.postprocess(highlight_mode=1)
    raw.postprocess(highlight_mode=rawpy.HighlightMode.Blend)
    raw.postprocess(highlight_mode=rawpy.HighlightMode.Reconstruct(3))

def testFindBadPixelsNikonD4():
    # crashed with "AssertionError: horizontal margins are not symmetric"
    find_bad_pixels([raw2TestPath])

def testNikonD4Size():
    if rawpy.libraw_version < (0,15):
        # older libraw/dcraw versions don't support D4 fully
        return
    raw = rawpy.imread(raw2TestPath)
    s = raw.sizes
    assert_equal(s.width, 4940)
    assert_equal(s.height, 3292)
    assert_equal(s.top_margin, 0)
    assert_equal(s.left_margin, 2)
    
def testSegfaultBug():
    # https://github.com/neothemachine/rawpy/issues/7
    im = rawpy.imread(rawTestPath).raw_image
    assert_array_equal(im.shape, [2844, 4288])
    print(im)

def testLibRawFileUnsupportedError():
    with pytest.raises(rawpy.LibRawFileUnsupportedError):
        rawpy.imread(os.path.join(thisDir, 'README.txt'))

def testLibRawOutOfOrderCallError():
    with pytest.raises(rawpy.LibRawOutOfOrderCallError):
        raw = rawpy.RawPy()
        raw.unpack()

def testCorruptFile():
    im = rawpy.imread(raw7TestPath)
    with pytest.raises(rawpy.LibRawDataError):
        im.postprocess()
    with pytest.raises(rawpy.LibRawDataError):
        im.extract_thumb()

def print_stats(rgb):
    print(rgb.dtype, 
          np.min(rgb, axis=(0,1)), np.max(rgb, axis=(0,1)), # range for each channel
          [len(np.unique(rgb[:,:,0])), len(np.unique(rgb[:,:,1])), len(np.unique(rgb[:,:,2]))], # unique values
          np.sum(rgb==np.iinfo(rgb.dtype).max, axis=(0,1))) # number of saturated pixels
        
if __name__ == '__main__':
    testVersion()
    #testFileOpenAndPostProcess()
    #testBadPixelRepair()
    testFindBadPixelsNikonD4()
    testCorruptFile()

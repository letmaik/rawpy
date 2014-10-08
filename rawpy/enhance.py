from __future__ import division, print_function, absolute_import

import time
import os
import warnings
import numpy as np

from skimage.filter.rank import median
try:
    import bottleneck
except ImportError:
    warnings.warn('"bottleneck" package not found, you cannot use method="mean" in repairBadPixels')

import rawpy

def findBadPixels(paths, find_hot=True, find_dead=True):
    assert find_hot or find_dead
    coords = []
    width = None
    for path in paths:
        t0 = time.time()
        raw = rawpy.imread(path)
        if width is None:
            # we need the width later for counting
            width = raw.raw_image.shape[1]
        print('imread:', time.time()-t0, 's')
    
        # TODO ignore border pixels
        
        # step 1: get color mask for each color
        color_masks = colormasks(raw)
        
        # step 2: median filtering for each channel        
        t0 = time.time()
        rawimg = raw.raw_image
        r = 5
        kernel = np.ones((r,r))
        thresh = max(np.max(rawimg)//150, 20)
        print('threshold:', thresh)
        for mask in color_masks:
            t1 = time.time()
            # skimage's median is quite slow, it uses an O(r) filtering algorithm.
            # There exist O(log(r)) and O(1) algorithms, see https://nomis80.org/ctmf.pdf.
            # Also, we only need the median values for the masked pixels.
            # Currently, they are calculated for all pixels for each color.
            med = median(rawimg, kernel, mask=mask)
            print('median:', time.time()-t1, 's')
            
            # step 3: detect possible bad pixels
            if find_hot and find_dead:
                candidates = (np.abs(rawimg - med) > thresh) & mask
            elif find_hot:
                candidates = (rawimg > med+thresh) & mask
            elif find_dead:
                candidates = (rawimg < med-thresh) & mask
            coords.append(np.transpose(np.nonzero(candidates)))
        print('badpixel candidates:', time.time()-t0, 's')
    
    # step 4: select candidates that appear on most input images
    # count how many times a coordinate appears
    coords = np.vstack(coords)
    
    # first we convert y,x to array offset such that we have an array of integers
    offset = coords[:,0]*width + coords[:,1]
    
    # now we count how many times each offset occurs
    counts = groupcount(offset)
    
    print('found', len(counts), 'bad pixel candidates, cross-checking images..')
    
    # we select the ones whose count is high
    is_bad = counts[:,1] >= 0.9*len(paths)
        
    # and convert back to y,x
    bad_offsets = counts[is_bad,0]
    bad_coords = np.transpose([bad_offsets // width, bad_offsets % width])
    
    print(len(bad_coords), 'bad pixels remaining after cross-checking images')
    
    return bad_coords
    
def mean(A, mask, kernel_size):
    """
    A faster alternative to skimage's mean filter.
    """
    dtype = A.dtype
    A = np.require(A, np.float)
    A[mask] = np.nan
    res = bottleneck.move_nanmean(A, kernel_size)
    np.round(res, out=res)
    return res.astype(dtype)

def repairBadPixels(raw, coords, method='mean'):
    print('repairing', len(coords), 'bad pixels')
    
    # TODO this can be done way more efficiently
    #  -> only interpolate at bad pixels instead of whole image
    #  -> cython? would likely involve for-loops
    #  see libraw/internal/dcraw_fileio.cpp
    
    t0 = time.time()
    
    color_masks = colormasks(raw)
        
    rawimg = raw.raw_image
    r = 5
    kernel = np.ones((r,r))
    for color_mask in color_masks:       
        mask = np.zeros_like(color_mask)
        mask[coords[:,0],coords[:,1]] = True
        mask &= color_mask
        
        # interpolate all bad pixels belonging to this color
        if method == 'mean':
            # FIXME could lead to invalid values if bad pixels are clustered
            smooth = mean(rawimg, mask, r)
        elif method == 'median':
            # bad pixels won't influence the median and just using
            # the color mask prevents bad pixel clusters from producing
            # bad interpolated values (NaNs)
            smooth = median(rawimg, kernel, mask=color_mask)
        else:
            raise ValueError
        
        rawimg[mask] = smooth[mask]
    
    print('badpixel repair:', time.time()-t0, 's')  
    
    # TODO check how many detected bad pixels are false positives
    #raw.raw_image[coords[:,0], coords[:,1]] = 0

def colormasks(raw):
    colors = raw.rawcolors
    if raw.num_colors == 3 and raw.color_desc == 'RGBG':
        color_masks = [colors == 0,
                       (colors == 1) | (colors == 3),
                       colors == 2]
    else:
        color_masks = [colors == i for i in range(len(raw.color_desc))]
    return color_masks

def groupcount(values):
    """
    :see: https://stackoverflow.com/a/4652265
    """
    values.sort()
    diff = np.concatenate(([1],np.diff(values)))
    idx = np.concatenate((np.where(diff)[0],[len(values)]))
    return np.transpose([values[idx[:-1]], np.diff(idx)])

if __name__ == '__main__':
    prefix = '../test/'
    testfiles = ['iss030e122639.NEF', 'iss030e122659.NEF', 'iss030e122679.NEF',
                 'iss030e122699.NEF', 'iss030e122719.NEF']
    paths = [prefix + f for f in testfiles]
    coords = findBadPixels(paths)
    print(coords)
    
    import imageio
    raw = rawpy.imread(paths[0])
    if not os.path.exists('test_original.png'):
        rgb = raw.postprocess()
        imageio.imsave('test_original.png', rgb)
    repairBadPixels(raw, coords)
    rgb = raw.postprocess()
    imageio.imsave('test_hotpixels_repaired.png', rgb)
    
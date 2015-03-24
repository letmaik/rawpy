from __future__ import division, print_function, absolute_import

"""
This module contains additional functionality not part of LibRaw.
"""

import time
import os
import warnings
from functools import partial
import numpy as np

from skimage.filter.rank import median
try:
    import cv2
except ImportError:
    warnings.warn('OpenCV not found, install for faster processing')
    cv2 = None

import rawpy

def _is_candidate(rawarr, med, find_hot, find_dead, thresh):
    if find_hot and find_dead:
        np.subtract(rawarr, med, out=med)
        np.abs(med, out=med)
        candidates = med > thresh
    elif find_hot:
        med += thresh
        candidates = rawarr > med
    elif find_dead:
        med -= thresh
        candidates = rawarr < med
    return candidates

def find_bad_pixels(paths, find_hot=True, find_dead=True, confirm_ratio=0.9):
    """
    Find and return coordinates of hot/dead pixels in the given RAW images.
    
    The probability that a detected bad pixel is really a bad pixel gets
    higher the more input images are given. The images should be taken around
    the same time, that is, each image must contain the same bad pixels.
    Also, there should be movement between the images to avoid the false detection
    of bad pixels in non-moving high-contrast areas.
    
    :param paths: paths to RAW images shot with the same camera
    :type paths: iterable of str
    :param bool find_hot: whether to find hot pixels
    :param bool find_dead: whether to find dead pixels
    :param float confirm_ratio: ratio of how many out of all given images
                          must contain a bad pixel to confirm it as such
    :return: coordinates of confirmed bad pixels
    :rtype: ndarray of shape (n,2) with y,x coordinates relative to visible RAW size
    """
    assert find_hot or find_dead
    coords = []
    width = None
    paths = list(paths)
    for path in paths:
        t0 = time.time()
        # TODO this is a bit slow, try RawSpeed
        raw = rawpy.imread(path)
        if width is None:
            if raw.raw_type != rawpy.RawType.Flat:
                raise NotImplementedError('Only Bayer-type images are currently supported')
            # we need the width later for counting
            width = raw.sizes.width
        print('imread:', time.time()-t0, 's')
            
        thresh = max(np.max(raw.raw_image_visible)//150, 20)
        print('threshold:', thresh)
        
        isCandidate = partial(_is_candidate, find_hot=find_hot, find_dead=find_dead, thresh=thresh)        
        coords.extend(_find_bad_pixel_candidates(raw, isCandidate))
    
    coords = np.vstack(coords)
    
    if len(paths) == 1:
        return coords
    
    # select candidates that appear on most input images
    # count how many times a coordinate appears
    
    # first we convert y,x to array offset such that we have an array of integers
    offset = coords[:,0]*width
    offset += coords[:,1]
    
    # now we count how many times each offset occurs
    t0 = time.time()
    counts = _groupcount(offset)
    print('groupcount:', time.time()-t0, 's')
    
    print('found', len(counts), 'bad pixel candidates, cross-checking images..')
    
    # we select the ones whose count is high
    is_bad = counts[:,1] >= confirm_ratio*len(paths)
        
    # and convert back to y,x
    bad_offsets = counts[is_bad,0]
    bad_coords = np.transpose([bad_offsets // width, bad_offsets % width])
    
    print(len(bad_coords), 'bad pixels remaining after cross-checking images')
    
    return bad_coords

def _find_bad_pixel_candidates(raw, isCandidateFn):
    t0 = time.time()
    
    if raw.raw_pattern.shape[0] == 2:
        coords = _find_bad_pixel_candidates_bayer2x2(raw, isCandidateFn)
    else:
        coords = _find_bad_pixel_candidates_generic(raw, isCandidateFn)
            
    print('badpixel candidates:', time.time()-t0, 's')
    
    return coords

def _find_bad_pixel_candidates_generic(raw, isCandidateFn):
    color_masks = _colormasks(raw)   
    rawimg = raw.raw_image_visible    
    coords = []
    
    # median filtering for each channel    
    r = 5
    kernel = np.ones((r,r))
    for mask in color_masks:
        t1 = time.time()
        # skimage's median is quite slow, it uses an O(r) filtering algorithm.
        # There exist O(log(r)) and O(1) algorithms, see https://nomis80.org/ctmf.pdf.
        # Also, we only need the median values for the masked pixels.
        # Currently, they are calculated for all pixels for each color.
        med = median(rawimg, kernel, mask=mask)
        print('median:', time.time()-t1, 's')
        
        # detect possible bad pixels
        candidates = isCandidateFn(rawimg, med)
        candidates &= mask
        
        y,x = np.nonzero(candidates)
        # note: the following is much faster than np.transpose((y,x))
        candidates = np.empty((len(y),2), dtype=y.dtype)
        candidates[:,0] = y
        candidates[:,1] = x
        
        coords.append(candidates)
        
    return coords

def _find_bad_pixel_candidates_bayer2x2(raw, isCandidateFn):
    assert raw.raw_pattern.shape[0] == 2
    
    # optimized code path for common 2x2 pattern
    # create a view for each color, do 3x3 median on it, find bad pixels, correct coordinates
    # This shortcut allows to do median filtering without using a mask, which means
    # that OpenCV's extremely fast median filter algorithm can be used.    
    r = 3
    
    rawimg = raw.raw_image_visible
    
    if cv2 is not None:
        median_ = partial(cv2.medianBlur, ksize=r)
    else:
        kernel = np.ones((r,r))
        median_ = partial(median, selem=kernel)
        
    coords = []
    
    # we have 4 colors (two greens are always seen as two colors)
    for offset_y in [0,1]:
        for offset_x in [0,1]:
            rawslice = rawimg[offset_y::2,offset_x::2]

            t1 = time.time()
            if cv2 is not None:
                # some older OpenCV versions require contiguous arrays
                # otherwise results are invalid
                rawslice = np.require(rawslice, rawslice.dtype, 'C')
            med = median_(rawslice)
            print('median:', time.time()-t1, 's')
            
            # detect possible bad pixels
            candidates = isCandidateFn(rawslice, med)
            
            # convert to coordinates and correct for slicing
            y,x = np.nonzero(candidates)
            # note: the following is much faster than np.transpose((y,x))
            candidates = np.empty((len(y),2), dtype=y.dtype)
            candidates[:,0] = y
            candidates[:,1] = x

            candidates *= 2
            candidates[:,0] += offset_y
            candidates[:,1] += offset_x
            
            coords.append(candidates)
            
    return coords

def repair_bad_pixels(raw, coords, method='median'):
    """
    Repair bad pixels in the given RAW image.
    
    Note that the function works in-place on the RAW image data.
    It has to be called before postprocessing the image.
    
    :param rawpy.RawPy raw: the RAW image to repair
    :param array-like coords: coordinates of bad pixels to repair, 
                           array of shape (n,2) with y,x coordinates relative to visible RAW size
    :param str method: currently only 'median' is supported
    """
    print('repairing', len(coords), 'bad pixels')
    
    # For small numbers of bad pixels this could be done more efficiently
    # by only interpolating the bad pixels instead of the whole image.
    
    coords = np.asarray(coords)
    
    t0 = time.time()
   
    if raw.raw_pattern.shape[0] == 2:
        _repair_bad_pixels_bayer2x2(raw, coords, method)
    else:
        _repair_bad_pixels_generic(raw, coords, method)
    
    print('badpixel repair:', time.time()-t0, 's')  
    
    # TODO check how many detected bad pixels are false positives
    #raw.raw_image_visible[coords[:,0], coords[:,1]] = 0

def _repair_bad_pixels_generic(raw, coords, method='median'):
    color_masks = _colormasks(raw)
        
    rawimg = raw.raw_image_visible
    r = 5
    kernel = np.ones((r,r))
    for color_mask in color_masks:       
        mask = np.zeros_like(color_mask)
        mask[coords[:,0],coords[:,1]] = True
        mask &= color_mask
        
        # interpolate all bad pixels belonging to this color
        if method == 'mean':
            # With mean filtering we have to ignore the bad pixels as they
            # would influence the mean too much.
            # FIXME could lead to invalid values if bad pixels are clustered
            #       such that no valid pixels are left in a block
            raise NotImplementedError
        elif method == 'median':
            # bad pixels won't influence the median in most cases and just using
            # the color mask prevents bad pixel clusters from producing
            # bad interpolated values (NaNs)
            smooth = median(rawimg, kernel, mask=color_mask)
        else:
            raise ValueError
        
        rawimg[mask] = smooth[mask]   
    
def _repair_bad_pixels_bayer2x2(raw, coords, method='median'):
    assert raw.raw_pattern.shape[0] == 2
    if method != 'median':
        raise NotImplementedError
    
    r = 3    
    rawimg = raw.raw_image_visible
    
    if cv2 is not None:
        median_ = partial(cv2.medianBlur, ksize=r)
    else:
        kernel = np.ones((r,r))
        median_ = partial(median, selem=kernel)
            
    # we have 4 colors (two greens are always seen as two colors)
    for offset_y in [0,1]:
        for offset_x in [0,1]:
            rawslice = rawimg[offset_y::2,offset_x::2]

            t1 = time.time()
            if cv2 is not None:
                # some older OpenCV versions require contiguous arrays
                # otherwise results are invalid
                rawslicecv = np.require(rawslice, rawslice.dtype, 'C')
                smooth = median_(rawslicecv)
            else:
                smooth = median_(rawslice)
            print('median:', time.time()-t1, 's')
            
            # determine which bad pixels belong to this color slice
            sliced_y = coords[:,0]-offset_y
            sliced_y %= 2
            sliced_x = coords[:,1]-offset_x
            sliced_x %= 2            
            matches_slice = sliced_y == 0
            matches_slice &= sliced_x == 0
                        
            coords_color = coords[matches_slice]
            
            # convert the full-size coordinates to the color slice coordinates
            coords_color[:,0] -= offset_y
            coords_color[:,1] -= offset_x
            coords_color /= 2            
            
            mask = np.zeros_like(rawslice, dtype=bool)
            mask[coords_color[:,0],coords_color[:,1]] = True

            rawslice[mask] = smooth[mask]

def _colormasks(raw):
    colors = raw.raw_colors_visible
    if raw.num_colors == 3 and raw.color_desc == 'RGBG':
        color_masks = [colors == 0,
                       (colors == 1) | (colors == 3),
                       colors == 2]
    else:
        color_masks = [colors == i for i in range(len(raw.color_desc))]
    return color_masks

def _groupcount(values):
    """
    :see: https://stackoverflow.com/a/4652265
    """
    values.sort()
    diff = np.concatenate(([1],np.diff(values)))
    idx = np.concatenate((np.where(diff)[0],[len(values)]))
    # note: the following is faster than np.transpose([vals,cnt])
    vals = values[idx[:-1]]
    cnt = np.diff(idx)
    res = np.empty((len(vals),2), dtype=vals.dtype)
    res[:,0] = vals
    res[:,1] = cnt
    return res

def save_dcraw_bad_pixels(path, bad_pixels):
    """
    Save the given bad pixel coordinates in 
    `dcraw file format <http://www.cybercom.net/~dcoffin/dcraw/.badpixels>`_.
    
    Note that timestamps cannot be set and will be written as zero.
    
    :param str path: path of the badpixels file that will be written
    :param array-like bad_pixels: array of shape (n,2) with y,x coordinates
                                  relative to visible image area
    """
    bad_pixels = np.asarray(bad_pixels)
    # Format is: pixel column, pixel row, UNIX time of death
    dc = np.zeros((len(bad_pixels),3), dtype=int)
    dc[:,0] = bad_pixels[:,1]
    dc[:,1] = bad_pixels[:,0]
    np.savetxt(path, dc, fmt='%1i')

if __name__ == '__main__':
    prefix = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test')
    testfiles = ['iss030e122639.NEF', 'iss030e122659.NEF', 'iss030e122679.NEF',
                 'iss030e122699.NEF', 'iss030e122719.NEF'][0:1]
    paths = [os.path.join(prefix, f) for f in testfiles]
    coords = find_bad_pixels(paths)
        
    import imageio
    raw = rawpy.imread(paths[0])
    if not os.path.exists('test_original.png'):
        rgb = raw.postprocess()
        imageio.imsave('test_original.png', rgb)
    
    # A. use dcraw repair
    # Note that this method fails when two bad pixels are direct neighbors.
    # This is because it corrects each bad pixel separately and uses the
    # mean of the surrounding pixels.
    t0 = time.time()
    bad_pixels_path = os.path.abspath('bad_pixels.txt')
    save_dcraw_bad_pixels(bad_pixels_path, coords)
    rgb = raw.postprocess(bad_pixels_path=bad_pixels_path)
    print('badpixel dcraw repair+postprocessing:', time.time()-t0, 's')
    imageio.imsave('test_hotpixels_repaired_dcraw.png', rgb)
    
    # B. use own repair function
    # With method='median' we still consider each bad pixel separately
    # but the median prevents neighboring bad pixels to have an influence.
    t0 = time.time()
    repair_bad_pixels(raw, coords, method='median')
    rgb = raw.postprocess()
    print('badpixel repair+postprocessing:', time.time()-t0, 's')
    imageio.imsave('test_hotpixels_repaired.png', rgb)
    
    # TODO method 'mean' not implemented yet
    
    

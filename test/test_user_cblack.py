"""Tests for per-channel black level corrections (user_cblack parameter)."""
from __future__ import division, print_function, absolute_import

import os
import pytest
import numpy as np
from numpy.testing import assert_array_equal

import rawpy

thisDir = os.path.dirname(__file__)
rawTestPath = os.path.join(thisDir, 'iss030e122639.NEF')


def test_user_cblack_parameter_acceptance():
    """Test that the user_cblack parameter is accepted in Params constructor."""
    # Test with valid 4-element list
    params = rawpy.Params(user_cblack=[100, 200, 150, 200])
    assert params.user_cblack == [100, 200, 150, 200]
    
    # Test with None (default)
    params = rawpy.Params()
    assert params.user_cblack == [0, 0, 0, 0]


def test_user_cblack_parameter_validation():
    """Test that user_cblack parameter validates list length."""
    # Should raise assertion error for wrong length
    with pytest.raises(AssertionError):
        rawpy.Params(user_cblack=[100, 200])
    
    with pytest.raises(AssertionError):
        rawpy.Params(user_cblack=[100, 200, 150])
    
    with pytest.raises(AssertionError):
        rawpy.Params(user_cblack=[100, 200, 150, 200, 250])


def test_user_cblack_postprocess():
    """Test that user_cblack can be used in postprocessing without errors."""
    with rawpy.imread(rawTestPath) as raw:
        # Process with per-channel black levels
        rgb = raw.postprocess(user_cblack=[100, 100, 100, 100], no_auto_bright=True)
        assert rgb.shape[2] == 3  # RGB image
        
        # Process with different per-channel values
        rgb2 = raw.postprocess(user_cblack=[50, 100, 150, 100], no_auto_bright=True)
        assert rgb2.shape[2] == 3
        
        # Images should be different when different black levels are applied
        assert not np.array_equal(rgb, rgb2)


def test_user_cblack_vs_user_black():
    """Test that user_cblack and user_black can be used together in a single call.
    
    user_cblack values are corrections/offsets applied on top of user_black.
    For example: user_black=100, user_cblack=[10, 20, 30, 20] results in
    effective black levels of [110, 120, 130, 120] for each channel.
    """
    with rawpy.imread(rawTestPath) as raw:
        # Process with both user_black and user_cblack in a single call
        # user_cblack provides per-channel corrections on top of user_black base value
        rgb = raw.postprocess(
            user_black=100,
            user_cblack=[10, 20, 30, 20],
            no_auto_bright=True
        )
        assert rgb.shape[2] == 3  # RGB image
        
        # Verify that using both parameters together works without errors
        # and produces a valid image
        assert rgb.dtype == np.uint8  # Default output_bps is 8

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

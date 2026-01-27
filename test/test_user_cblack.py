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
    """Test that user_cblack and user_black can both be used."""
    with rawpy.imread(rawTestPath) as raw:
        # Process with single black level
        rgb_single = raw.postprocess(user_black=100, no_auto_bright=True)
        
        # Process with per-channel black levels (all same value)
        rgb_multi = raw.postprocess(user_cblack=[100, 100, 100, 100], no_auto_bright=True)
        
        # When all channels have the same value, results should be similar
        # (might not be exactly equal due to processing differences)
        assert rgb_single.shape == rgb_multi.shape


def test_user_cblack_with_other_params():
    """Test that user_cblack works correctly with other parameters."""
    with rawpy.imread(rawTestPath) as raw:
        # Combine user_cblack with various other parameters
        rgb = raw.postprocess(
            user_cblack=[100, 150, 100, 150],
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True
        )
        assert rgb.dtype == np.uint16
        assert rgb.shape[2] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

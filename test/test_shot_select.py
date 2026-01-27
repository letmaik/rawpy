"""
Test for shot_select parameter functionality
"""
import os
import pytest
import rawpy

thisDir = os.path.dirname(__file__)
rawTestPath = os.path.join(thisDir, 'iss030e122639.NEF')


def test_shot_select_parameter_exists():
    """Test that shot_select parameter can be passed to postprocess"""
    with rawpy.imread(rawTestPath) as raw:
        # Test default shot_select=0
        rgb = raw.postprocess(shot_select=0, no_auto_bright=True)
        assert rgb is not None
        assert rgb.shape[2] == 3  # RGB image
        

def test_shot_select_different_values():
    """Test that different shot_select values can be set"""
    with rawpy.imread(rawTestPath) as raw:
        # Test shot_select=0
        rgb0 = raw.postprocess(shot_select=0, no_auto_bright=True)
        assert rgb0 is not None
        
        # Test shot_select=1 (may not have effect on single-shot files)
        rgb1 = raw.postprocess(shot_select=1, no_auto_bright=True)
        assert rgb1 is not None
        
        # Both should produce valid images (same for single-shot files)
        assert rgb0.shape == rgb1.shape


def test_shot_select_via_params():
    """Test that shot_select can be passed via Params object"""
    with rawpy.imread(rawTestPath) as raw:
        params = rawpy.Params(shot_select=0)
        rgb = raw.postprocess(params=params)
        assert rgb is not None
        assert rgb.shape[2] == 3


def test_shot_select_default_value():
    """Test that default shot_select value is 0"""
    params = rawpy.Params()
    assert params.shot_select == 0

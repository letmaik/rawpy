"""
Test for shot_select parameter functionality
"""
import os
import pytest
import rawpy

thisDir = os.path.dirname(__file__)
rawTestPath = os.path.join(thisDir, 'iss030e122639.NEF')


def test_shot_select_parameter_via_imread():
    """Test that shot_select parameter can be passed to imread"""
    # Test default shot_select=0
    with rawpy.imread(rawTestPath, shot_select=0) as raw:
        rgb = raw.postprocess(no_auto_bright=True)
        assert rgb is not None
        assert rgb.shape[2] == 3  # RGB image


def test_shot_select_nonexistent_image():
    """Test that shot_select=1 raises error for single-image files"""
    # Test shot_select=1 on a file with only one image should raise an error
    with pytest.raises(rawpy.LibRawRequestForNonexistentImageError):
        with rawpy.imread(rawTestPath, shot_select=1) as raw:
            rgb = raw.postprocess(no_auto_bright=True)


def test_shot_select_via_open_file():
    """Test that shot_select can be set via set_unpack_params"""
    raw = rawpy.RawPy()
    raw.open_file(rawTestPath)
    raw.set_unpack_params(shot_select=0)
    raw.unpack()
    rgb = raw.postprocess()
    assert rgb is not None
    assert rgb.shape[2] == 3
    raw.close()


def test_shot_select_default_value():
    """Test that default shot_select value is 0"""
    with rawpy.imread(rawTestPath) as raw:
        rgb = raw.postprocess()
        assert rgb is not None
        assert rgb.shape[2] == 3

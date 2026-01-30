"""
Test that type checkers can see all the types exported by rawpy.

This validates that the TYPE_CHECKING imports in __init__.py include
all the types that are available at runtime via globals().update().
"""

import rawpy

def test_runtime_imports_available() -> None:
    """Test that all expected types are available at runtime."""
    # These should all be available at runtime due to globals().update()
    assert hasattr(rawpy, 'RawPy')
    assert hasattr(rawpy, 'Params')
    assert hasattr(rawpy, 'ImageSizes')
    assert hasattr(rawpy, 'Thumbnail')
    assert hasattr(rawpy, 'RawType')
    assert hasattr(rawpy, 'ThumbFormat')
    assert hasattr(rawpy, 'DemosaicAlgorithm')
    assert hasattr(rawpy, 'ColorSpace')
    assert hasattr(rawpy, 'HighlightMode')
    assert hasattr(rawpy, 'FBDDNoiseReductionMode')
    # Exceptions
    assert hasattr(rawpy, 'LibRawError')
    assert hasattr(rawpy, 'LibRawIOError')

def test_type_checker_sees_types() -> None:
    """
    Test that type checkers can see the types.
    
    If mypy can type check this function without errors, then the types
    are properly imported in the TYPE_CHECKING block.
    """
    # These type annotations should be recognized by mypy
    sizes: rawpy.ImageSizes
    thumb: rawpy.Thumbnail  
    raw_type: rawpy.RawType
    thumb_fmt: rawpy.ThumbFormat
    algo: rawpy.DemosaicAlgorithm
    color: rawpy.ColorSpace
    highlight: rawpy.HighlightMode
    noise: rawpy.FBDDNoiseReductionMode
    params: rawpy.Params
    error: rawpy.LibRawError
    
    # Use them to avoid unused variable warnings
    _ = (sizes, thumb, raw_type, thumb_fmt, algo, color, highlight, noise, params, error)

if __name__ == "__main__":
    print("Testing runtime imports...")
    test_runtime_imports_available()
    print("✓ All runtime imports available")
    
    print("\nTesting type checker visibility...")
    test_type_checker_sees_types()
    print("✓ Type checker test passed")

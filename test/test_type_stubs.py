"""Test script to verify type hints work correctly with type checkers like mypy and pyright."""

import rawpy
import numpy as np


def test_basic_types() -> None:
    """Test basic type hints for rawpy module."""
    # Test imread with string path
    raw: rawpy.RawPy = rawpy.imread("test.nef")
    
    # Test context manager
    with rawpy.imread("test.nef") as raw:
        # Test properties
        sizes: rawpy.ImageSizes = raw.sizes
        width: int = sizes.width
        height: int = sizes.height
        
        # Test raw_image property
        img: np.ndarray = raw.raw_image
        
        # Test raw_image_visible property
        visible: np.ndarray = raw.raw_image_visible
        
        # Test white_level property
        white: int = raw.white_level
        
        # Test camera_whitebalance property
        wb: list[float] = raw.camera_whitebalance
        
        # Test postprocess with kwargs
        rgb1: np.ndarray = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            gamma=(1.0, 1.0),
            output_bps=16,
        )
        
        # Test postprocess with Params
        params = rawpy.Params(
            use_camera_wb=True,
            output_bps=16,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
        rgb2: np.ndarray = raw.postprocess(params)
        
        # Test extract_thumb with proper type narrowing
        thumb: rawpy.Thumbnail = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            # Type narrowing: when format is JPEG, data should be bytes
            jpeg_data: bytes = thumb.data if isinstance(thumb.data, bytes) else b''
        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            # Type narrowing: when format is BITMAP, data should be ndarray
            bitmap_data: np.ndarray = thumb.data if isinstance(thumb.data, np.ndarray) else np.array([])


def test_enums() -> None:
    """Test enum type hints."""
    # Test DemosaicAlgorithm
    algo: rawpy.DemosaicAlgorithm = rawpy.DemosaicAlgorithm.AHD
    
    # Test ColorSpace
    color: rawpy.ColorSpace = rawpy.ColorSpace.sRGB
    
    # Test HighlightMode
    mode: rawpy.HighlightMode = rawpy.HighlightMode.Clip
    
    # Test RawType
    raw_type: rawpy.RawType = rawpy.RawType.Flat


def test_exceptions() -> None:
    """Test exception type hints."""
    try:
        rawpy.imread("nonexistent.nef")
    except rawpy.LibRawFileUnsupportedError as e:
        print(f"File unsupported: {e}")
    except rawpy.LibRawError as e:
        print(f"LibRaw error: {e}")


if __name__ == "__main__":
    print("This test file is for type checking only.")
    print("Run with: mypy test_type_stubs.py or pyright test_type_stubs.py")

"""Test script to verify inline type hints work correctly with type checkers like mypy and pyright."""

import rawpy
import numpy as np


def test_basic_types() -> None:
    """Test basic type hints for rawpy module."""
    # Test imread with string path
    raw: rawpy.RawPy = rawpy.imread("test.nef")
    
    # Test context manager
    with rawpy.imread("test.nef") as raw:
        # Test properties - note: these would need runtime to actually access
        # but type checker should validate the annotations
        pass
        
        # These would require the Cython module to provide type hints
        # For now, the type checker will use 'Any' for attributes from the Cython module


def test_imread_signature() -> None:
    """Test that imread accepts both string and file-like objects."""
    # String path
    raw1: rawpy.RawPy = rawpy.imread("test.nef")
    
    # File-like object (BinaryIO)
    from io import BytesIO
    file_obj = BytesIO()
    raw2: rawpy.RawPy = rawpy.imread(file_obj)


if __name__ == "__main__":
    print("This test file is for type checking only.")
    print("Run with: mypy test_type_stubs.py or pyright test_type_stubs.py")


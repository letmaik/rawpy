"""Test script to verify inline type hints work correctly with type checkers like mypy and pyright."""

import os
import rawpy
import numpy as np


# Path to existing test image
thisDir = os.path.dirname(__file__)
testImagePath = os.path.join(thisDir, 'iss030e122639.NEF')


def test_basic_types() -> None:
    """Test basic type hints for rawpy module."""
    # Test imread with string path
    raw: rawpy.RawPy = rawpy.imread(testImagePath)
    
    # Test context manager
    with rawpy.imread(testImagePath) as raw:
        # Test properties - note: these would need runtime to actually access
        # but type checker should validate the annotations
        pass
        
        # These would require the Cython module to provide type hints
        # For now, the type checker will use 'Any' for attributes from the Cython module


def test_imread_signature() -> None:
    """Test that imread accepts both string and file-like objects."""
    # String path
    raw1: rawpy.RawPy = rawpy.imread(testImagePath)
    
    # File-like object (BinaryIO)
    from io import BytesIO
    with open(testImagePath, 'rb') as f:
        file_obj = BytesIO(f.read())
    raw2: rawpy.RawPy = rawpy.imread(file_obj)


if __name__ == "__main__":
    print("This test file is for type checking only.")
    print("Run with: mypy test_type_stubs.py or pyright test_type_stubs.py")

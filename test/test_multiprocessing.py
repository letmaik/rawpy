"""
Test for multiprocessing with rawpy to ensure no deadlocks occur.
"""
from __future__ import division, print_function, absolute_import

import os
import sys
import multiprocessing as mp
import pytest
import warnings

import rawpy

thisDir = os.path.dirname(__file__)

# Use a test RAW file that exists
rawTestPath = os.path.join(thisDir, 'iss030e122639.NEF')


def load_and_process_raw(filepath):
    """Function to be executed in child process."""
    # This should work without deadlocking when using 'spawn' method
    with rawpy.imread(filepath) as raw:
        rgb = raw.postprocess(no_auto_bright=True, half_size=True)
    return rgb.shape


def test_multiprocessing_spawn():
    """Test that multiprocessing works with 'spawn' method."""
    # Skip on Windows where fork is not the default
    if sys.platform == 'win32':
        pytest.skip("Test only relevant on Unix-like systems")
    
    # Get current start method
    original_method = mp.get_start_method(allow_none=True)
    
    try:
        # Set start method to 'spawn' - this is the recommended way
        # Note: This might fail if context has already been set
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, use context instead
            ctx = mp.get_context('spawn')
            with ctx.Pool(processes=2) as pool:
                results = pool.map(load_and_process_raw, [rawTestPath, rawTestPath])
            assert len(results) == 2
            for shape in results:
                assert len(shape) == 3  # (height, width, channels)
            return
        
        # Use multiprocessing with spawn
        with mp.Pool(processes=2) as pool:
            results = pool.map(load_and_process_raw, [rawTestPath, rawTestPath])
        
        assert len(results) == 2
        for shape in results:
            assert len(shape) == 3  # (height, width, channels)
            
    finally:
        # Try to restore original method (may not work, but try anyway)
        if original_method:
            try:
                mp.set_start_method(original_method, force=True)
            except RuntimeError:
                pass


def test_multiprocessing_warning_detection():
    """Test that the warning detection function exists and works correctly."""
    # Skip on Windows
    if sys.platform == 'win32':
        pytest.skip("Test only relevant on Unix-like systems")
    
    # Only test if OpenMP is enabled
    if not rawpy.flags or not rawpy.flags.get('OPENMP', False):  # type: ignore[attr-defined]
        pytest.skip("OpenMP not enabled, warning not expected")
    
    # Verify the warning function exists
    from rawpy import _check_multiprocessing_fork
    assert _check_multiprocessing_fork is not None
    
    # When called in main process, should not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_multiprocessing_fork()
        # Should not produce warning in main process
        fork_warnings = [warning for warning in w if 'fork' in str(warning.message).lower()]
        assert len(fork_warnings) == 0


if __name__ == '__main__':
    print("Testing multiprocessing with spawn method...")
    test_multiprocessing_spawn()
    print("SUCCESS: No deadlocks with spawn method!")
    
    print("\nTesting warning detection...")
    test_multiprocessing_warning_detection()
    print("SUCCESS: Warning system working correctly!")

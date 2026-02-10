"""
Test that the .pyi stub file matches the runtime signatures of the _rawpy module.

This test uses mypy's stubtest tool, which is the industry standard for validating
that stub files accurately reflect runtime module signatures.

Note: This test requires the rawpy module to be built and installed/importable.
"""

import subprocess
import sys
import os
import pytest
import rawpy

# stubtest validates that the .pyi stub matches the runtime module.
# When rawpy is installed from an artifact (site-packages), skip — the
# allowlist and stub source are tied to the editable/source-tree workflow.
# Note: .venv-test is inside the repo root, so we compare against rawpy/ subdir.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_rawpy_dir = os.path.join(os.path.abspath(_repo_root), "rawpy")
_is_editable = os.path.abspath(rawpy.__file__).startswith(_rawpy_dir)


@pytest.mark.skipif(not _is_editable, reason="requires editable install")
def test_stub_matches_runtime():
    """
    Use mypy stubtest to verify that rawpy/_rawpy.pyi matches the runtime signatures.
    
    stubtest compares the stub file against the actual runtime using Python's inspect
    module, checking for:
    - Missing or extra functions/methods/properties
    - Signature mismatches (parameters, return types)
    - Missing or extra class members
    
    This is the recommended approach from the Python typing community and is used
    by typeshed to validate all their stubs.
    
    Internal implementation details (Cython-generated methods, internal classes)
    are excluded via the stubtest_allowlist.txt file.
    """
    # Import the module - this will fail if not built
    import rawpy._rawpy
    
    # Get path to allowlist file
    test_dir = os.path.dirname(__file__)
    allowlist_path = os.path.join(test_dir, 'stubtest_allowlist.txt')
    
    # Run stubtest on the _rawpy module with allowlist
    # The stub file rawpy/_rawpy.pyi will be automatically found by mypy
    # Use --ignore-disjoint-bases to suppress disjoint base warnings for Cython classes
    # (these are internal implementation details not relevant for type checking)
    result = subprocess.run(
        [sys.executable, "-m", "mypy.stubtest", "rawpy._rawpy", 
         "--allowlist", allowlist_path,
         "--ignore-disjoint-bases"],
        capture_output=True,
        text=True
    )
    
    # Check if stubtest command exists
    if "No module named mypy.stubtest" in result.stderr or "No module named mypy" in result.stderr:
        pytest.fail("mypy is not installed. Install with: pip install mypy")
    
    # If there are mismatches, stubtest will return non-zero and output details
    if result.returncode != 0:
        error_msg = f"""
Stub file (rawpy/_rawpy.pyi) does not match runtime signatures!

STDOUT:
{result.stdout}

STDERR:
{result.stderr}

To fix this, update rawpy/_rawpy.pyi to match the runtime signatures.
To run stubtest manually: python -m mypy.stubtest rawpy._rawpy
"""
        pytest.fail(error_msg)
    
    # Success - stubs match runtime
    assert result.returncode == 0, "Stubtest should pass with no mismatches"


if __name__ == "__main__":
    # Allow running the test directly for debugging
    test_stub_matches_runtime()
    print("✓ Stub file matches runtime signatures!")

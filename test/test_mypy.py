"""
Test that mypy type checking passes for rawpy package and tests.

This validates that:
- Type annotations in rawpy module are correct
- Test files use types correctly
- No mypy errors in the codebase
"""

import subprocess
import sys
import os


def test_mypy_all():
    """
    Run mypy on both rawpy/ package and test/ directory to validate type annotations.
    
    This ensures that:
    - All type annotations in the package are correct and internally consistent
    - Test files properly use the type-annotated rawpy API
    """
    # Check if mypy is installed
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError("mypy is not installed. Install with: pip install mypy")
    
    # Build absolute paths based on test file location
    test_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(test_dir)
    rawpy_dir = os.path.join(repo_root, "rawpy")
    test_path = test_dir
    mypy_config = os.path.join(repo_root, "mypy.ini")
    
    # Run mypy on both rawpy package and test directory at once
    # Use --install-types to automatically install missing type stubs
    # Use --non-interactive to avoid prompts in CI
    # Config file selectively ignores optional dependencies (skimage, cv2)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", 
         "--install-types", "--non-interactive",
         "--config-file", mypy_config,
         rawpy_dir, test_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        error_msg = f"""
mypy found type errors!

STDOUT:
{result.stdout}

STDERR:
{result.stderr}

To fix this, address the type errors shown above.
To run mypy manually: python -m mypy --install-types --non-interactive --config-file mypy.ini rawpy/ test/
"""
        raise AssertionError(error_msg)
    
    # Success
    assert result.returncode == 0, "mypy should pass with no errors"


if __name__ == "__main__":
    # Allow running the test directly for debugging
    print("Running mypy on rawpy/ and test/ ...")
    test_mypy_all()
    print("✓ mypy passed on all checked directories!")


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


def test_mypy_rawpy_package():
    """
    Run mypy on the rawpy package to validate type annotations.
    
    This ensures that all type annotations in the package are correct
    and internally consistent.
    """
    # Check if mypy is installed
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError("mypy is not installed. Install with: pip install mypy")
    
    # Run mypy on rawpy package
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "rawpy/"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    
    if result.returncode != 0:
        error_msg = f"""
mypy found type errors in rawpy/ package!

STDOUT:
{result.stdout}

STDERR:
{result.stderr}

To fix this, address the type errors shown above.
To run mypy manually: python -m mypy rawpy/
"""
        raise AssertionError(error_msg)
    
    # Success
    assert result.returncode == 0, "mypy should pass with no errors"


def test_mypy_tests():
    """
    Run mypy on the test directory to validate test code uses types correctly.
    
    This ensures that test files properly use the type-annotated rawpy API.
    """
    # Check if mypy is installed
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError("mypy is not installed. Install with: pip install mypy")
    
    # Run mypy on test directory
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "test/"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    
    if result.returncode != 0:
        error_msg = f"""
mypy found type errors in test/ directory!

STDOUT:
{result.stdout}

STDERR:
{result.stderr}

To fix this, address the type errors shown above.
To run mypy manually: python -m mypy test/
"""
        raise AssertionError(error_msg)
    
    # Success
    assert result.returncode == 0, "mypy should pass with no errors"


if __name__ == "__main__":
    # Allow running the tests directly for debugging
    print("Running mypy on rawpy/ package...")
    test_mypy_rawpy_package()
    print("✓ mypy passed on rawpy/ package")
    
    print("\nRunning mypy on test/ directory...")
    test_mypy_tests()
    print("✓ mypy passed on test/ directory")
    
    print("\n✓ All mypy checks passed!")

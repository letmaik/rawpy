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
import pytest
import rawpy

# These tests type-check the source tree (rawpy/ and test/ at repo root).
# When rawpy is installed from an artifact (site-packages), skip — the source
# tree's rawpy/ would shadow or conflict with the installed package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_is_editable = os.path.abspath(rawpy.__file__).startswith(os.path.abspath(_repo_root))


@pytest.mark.skipif(not _is_editable, reason="requires editable install")
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
    
    # Get repo root from test file location
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run mypy on both rawpy package and test directory at once
    # Use --install-types to automatically install missing type stubs
    # Use --non-interactive to avoid prompts in CI
    # Config is in pyproject.toml [tool.mypy], auto-discovered via cwd
    result = subprocess.run(
        [sys.executable, "-m", "mypy", 
         "--install-types", "--non-interactive",
         "rawpy/", "test/"],
        capture_output=True,
        text=True,
        cwd=repo_root
    )
    
    if result.returncode != 0:
        error_msg = f"""
mypy found type errors!

STDOUT:
{result.stdout}

STDERR:
{result.stderr}

To fix this, address the type errors shown above.
To run mypy manually: python -m mypy --install-types --non-interactive rawpy/ test/
"""
        raise AssertionError(error_msg)
    
    # Success
    assert result.returncode == 0, "mypy should pass with no errors"


if __name__ == "__main__":
    # Allow running the test directly for debugging
    print("Running mypy on rawpy/ and test/ ...")
    test_mypy_all()
    print("✓ mypy passed on all checked directories!")


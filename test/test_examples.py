"""
Test that example scripts run without errors.

Each example is executed as a subprocess so it runs exactly as a user would.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
TEST_IMAGE = os.path.join(REPO_ROOT, "test", "iss030e122639.NEF")

needs_test_image = pytest.mark.skipif(
    not os.path.exists(TEST_IMAGE),
    reason="test image not available",
)

skip_examples = pytest.mark.skipif(
    os.environ.get("RAWPY_SKIP_EXAMPLES", "") == "1",
    reason="RAWPY_SKIP_EXAMPLES is set (e.g., slow QEMU emulation)",
)

pytestmark = skip_examples


def run_example(script_name: str) -> subprocess.CompletedProcess:
    """Run an example script and return the result."""
    script_path = os.path.join(EXAMPLES_DIR, script_name)
    return subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=120,
    )


@needs_test_image
def test_basic_process():
    result = run_example("basic_process.py")
    assert result.returncode == 0, result.stderr


@needs_test_image
def test_thumbnail_extract():
    result = run_example("thumbnail_extract.py")
    assert result.returncode == 0, result.stderr


@needs_test_image
def test_bad_pixel_repair():
    result = run_example("bad_pixel_repair.py")
    assert result.returncode == 0, result.stderr

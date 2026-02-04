"""
Basic RAW Processing Example

Demonstrates:
- Loading a RAW file with rawpy.imread()
- Converting to RGB with postprocess()
- Saving the result

Usage:
    python examples/basic_process.py
"""

import rawpy
import imageio.v3 as iio
import os
import sys
import tempfile

# Locate test image (works from repo root or examples/ directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_IMAGE = os.path.join(REPO_ROOT, "test", "iss030e122639.NEF")


def main():
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found at {TEST_IMAGE}")
        print("This example requires the test data from the repository.")
        return 1

    print(f"Loading: {TEST_IMAGE}")

    with rawpy.imread(TEST_IMAGE) as raw:
        print(f"  Raw type: {raw.raw_type}")
        print(f"  Image size: {raw.sizes.width}x{raw.sizes.height}")

        # Convert RAW to RGB using default parameters
        rgb = raw.postprocess()
        print(f"  Output shape: {rgb.shape}")

        # Save to temp directory (avoids polluting repo)
        output_path = os.path.join(tempfile.gettempdir(), "rawpy_basic_output.tiff")
        iio.imwrite(output_path, rgb)
        print(f"  Saved to: {output_path}")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

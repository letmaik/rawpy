"""
Bad Pixel Repair Example

Demonstrates:
- Using rawpy.enhance module for bad pixel detection/repair
- Loading pre-computed bad pixel coordinates
- Repairing bad pixels using median interpolation

Note: In practice, you would first detect bad pixels using:
    bad_pixels = rawpy.enhance.find_bad_pixels(['image1.NEF', 'image2.NEF', ...])

Usage:
    python examples/bad_pixel_repair.py
"""

import numpy as np
import rawpy
import rawpy.enhance
import imageio.v3 as iio
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_IMAGE = os.path.join(REPO_ROOT, "test", "iss030e122639.NEF")
BAD_PIXELS_FILE = os.path.join(REPO_ROOT, "test", "bad_pixels.gz")


def main():
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found at {TEST_IMAGE}")
        return 1

    # Load bad pixel coordinates (if available)
    if not os.path.exists(BAD_PIXELS_FILE):
        print(f"Bad pixel file not found: {BAD_PIXELS_FILE}")
        print("Skipping repair demo. In practice, you would run:")
        print("  bad_pixels = rawpy.enhance.find_bad_pixels([...image paths...])")
        return 0

    bad_pixels = np.loadtxt(BAD_PIXELS_FILE, dtype=int)

    print(f"Loaded {len(bad_pixels)} bad pixel coordinates")
    print(f"Processing: {TEST_IMAGE}")

    with rawpy.imread(TEST_IMAGE) as raw:
        # Repair bad pixels in-place before postprocessing
        rawpy.enhance.repair_bad_pixels(raw, bad_pixels, method="median")

        # Now postprocess the repaired data
        rgb = raw.postprocess()

        output_path = os.path.join(tempfile.gettempdir(), "rawpy_repaired.tiff")
        iio.imwrite(output_path, rgb)
        print(f"Saved repaired image to: {output_path}")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

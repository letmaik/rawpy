"""
Thumbnail Extraction Example

Demonstrates:
- Extracting embedded JPEG thumbnails from RAW files
- Handling different thumbnail formats (JPEG vs BITMAP)
- Error handling for missing/unsupported thumbnails

Usage:
    python examples/thumbnail_extract.py
"""

import rawpy
import imageio.v3 as iio
import os
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TEST_IMAGE = os.path.join(REPO_ROOT, "test", "iss030e122639.NEF")


def main():
    if not os.path.exists(TEST_IMAGE):
        print(f"Error: Test image not found at {TEST_IMAGE}")
        return 1

    print(f"Extracting thumbnail from: {TEST_IMAGE}")

    with rawpy.imread(TEST_IMAGE) as raw:
        try:
            thumb = raw.extract_thumb()
        except rawpy.LibRawNoThumbnailError:
            print("No thumbnail embedded in this file.")
            return 0
        except rawpy.LibRawUnsupportedThumbnailError:
            print("Thumbnail format not supported.")
            return 0

        print(f"  Thumbnail format: {thumb.format}")

        output_dir = tempfile.gettempdir()

        if thumb.format == rawpy.ThumbFormat.JPEG:
            output_path = os.path.join(output_dir, "rawpy_thumb.jpg")
            with open(output_path, "wb") as f:
                f.write(thumb.data)
            print(f"  Saved JPEG to: {output_path}")

        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            output_path = os.path.join(output_dir, "rawpy_thumb.tiff")
            iio.imwrite(output_path, thumb.data)
            print(f"  Saved TIFF to: {output_path}")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Test that rawpy feature flags are present and that CI wheels have all
expected features enabled.

The CI build scripts install zlib, libjpeg-turbo (jpeg8), libjasper,
and lcms2 on all platforms, so all codec-related flags should be True
in CI-built wheels.

When running locally (editable install), some flags may be False because
the system libraries are not installed. The test for CI-required flags
is skipped in that case.
"""

import os

import pytest
import rawpy


@pytest.mark.skipif(
    rawpy.flags is None,
    reason="libraw_config.h not available (non-CMake LibRaw build)",
)
def test_flags_present():
    """rawpy.flags should be a dict (not None) with all known keys."""
    assert rawpy.flags is not None, (
        "rawpy.flags is None — libraw_config.h was not found at build time"
    )
    expected_keys = {
        "DNGDEFLATECODEC",
        "DNGLOSSYCODEC",
        "OPENMP",
        "LCMS",
        "REDCINECODEC",
        "RAWSPEED",
        "DEMOSAIC_PACK_GPL2",
        "DEMOSAIC_PACK_GPL3",
        "X3FTOOLS",
        "6BY9RPI",
    }
    assert set(rawpy.flags.keys()) == expected_keys, (
        f"Unexpected flag keys: {set(rawpy.flags.keys()) ^ expected_keys}"
    )


@pytest.mark.skipif(
    rawpy.flags is None,
    reason="libraw_config.h not available (non-CMake LibRaw build)",
)
def test_flags_are_bool():
    """All flag values should be booleans."""
    assert rawpy.flags is not None
    for key, value in rawpy.flags.items():
        assert isinstance(value, bool), f"flags[{key!r}] = {value!r}, expected bool"



@pytest.mark.skipif(
    rawpy.flags is None,
    reason="libraw_config.h not available (non-CMake LibRaw build)",
)
@pytest.mark.skipif(
    "CI" not in os.environ,
    reason="only enforced in CI where all dependencies are installed",
)
def test_wheel_feature_flags():
    """CI wheels must have all expected feature flags enabled."""
    assert rawpy.flags is not None

    # Flags that must be True in every CI wheel.
    required_true = {
        "DNGDEFLATECODEC",  # zlib
        "DNGLOSSYCODEC",    # libjpeg-turbo (jpeg8)
        "REDCINECODEC",     # libjasper
        "LCMS",             # lcms2
        "X3FTOOLS",         # always enabled in setup.py
        "6BY9RPI",          # always enabled in setup.py
    }

    # Flags that must be False (not enabled / not bundled).
    required_false = {
        "RAWSPEED",         # never enabled
        "DEMOSAIC_PACK_GPL2",  # GPL, not bundled in MIT wheels
        "DEMOSAIC_PACK_GPL3",  # GPL, not bundled in MIT wheels
    }

    # OpenMP: enabled on Windows (when VC runtime DLL found), disabled on
    # Unix. We don't assert it here since it varies by platform.

    errors = []
    for flag in required_true:
        if not rawpy.flags.get(flag):
            errors.append(f"{flag} should be True but is {rawpy.flags.get(flag)}")
    for flag in required_false:
        if rawpy.flags.get(flag):
            errors.append(f"{flag} should be False but is {rawpy.flags.get(flag)}")

    assert not errors, "Feature flag mismatches:\n" + "\n".join(f"  - {e}" for e in errors)


def test_libraw_version():
    """rawpy.libraw_version should be a tuple of three ints >= 0.21."""
    ver = rawpy.libraw_version
    assert isinstance(ver, tuple), f"Expected tuple, got {type(ver)}"
    assert len(ver) == 3, f"Expected 3 elements, got {len(ver)}"
    assert all(isinstance(v, int) for v in ver), f"Expected ints, got {ver}"
    assert ver >= (0, 21, 0), f"LibRaw version {ver} is older than 0.21.0"

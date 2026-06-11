"""gasgiant.palette.reference: quartile latitude profiles."""

from __future__ import annotations

import numpy as np

from gasgiant.palette.reference import latitude_profile, profile_distance


def _banded_image(h: int = 128, w: int = 256) -> np.ndarray:
    """Alternating bright/dark rows with a sprinkle of outlier pixels."""
    img = np.zeros((h, w, 3), dtype=np.float32)
    bright = (0.9, 0.85, 0.7)
    dark = (0.45, 0.3, 0.15)
    for row in range(h):
        img[row] = bright if (row // 16) % 2 == 0 else dark
    # Outliers a mean would smear into the profile.
    img[:, ::37] = (1.0, 0.0, 0.0)
    return img


def test_latitude_axis_orientation():
    p = latitude_profile(_banded_image(), bins=32)
    assert p.lat_deg[0] > 80.0
    assert p.lat_deg[-1] < -80.0
    assert np.all(np.diff(p.lat_deg) < 0)


def test_quartile_decomposition_resists_outliers():
    # Within a bin spanning both band values, zone tracks the bright band and
    # belt the dark band; the red outlier columns must not bend the medians.
    p = latitude_profile(_banded_image(), bins=4)
    assert np.allclose(p.zone_rgb, (0.9, 0.85, 0.7), atol=0.02)
    assert np.allclose(p.belt_rgb, (0.45, 0.3, 0.15), atol=0.02)


def test_contrast_envelope_positive_on_banded_input():
    p = latitude_profile(_banded_image(), bins=4)
    assert np.all(p.contrast > 0.2)
    flat = latitude_profile(np.full((64, 128, 3), 0.5, np.float32), bins=4)
    assert np.all(flat.contrast < 1e-3)


def test_profile_distance_zero_on_self():
    p = latitude_profile(_banded_image(), bins=16)
    d = profile_distance(p, p)
    assert all(v == 0.0 for v in d.values())


def test_profile_distance_resamples_bin_counts():
    img = _banded_image()
    a = latitude_profile(img, bins=16)
    b = latitude_profile(img, bins=48)
    d = profile_distance(a, b)
    assert all(v < 0.05 for v in d.values())

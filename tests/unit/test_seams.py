from __future__ import annotations

import numpy as np

from gasgiant.core.domain import EquirectGrid
from gasgiant.validate import validate_arrays
from gasgiant.validate.seams import Report, check_pole_rows, check_wrap_continuity


def _smooth_sphere_map(width=256):
    """A seam-free map: smooth function of the 3D sphere position."""
    grid = EquirectGrid(width, width // 2)
    pts = grid.sphere_points().astype(np.float64)
    return (
        0.5
        + 0.2 * np.sin(3.0 * pts[..., 0] + 2.0 * pts[..., 2])
        + 0.2 * np.sin(4.0 * pts[..., 1])
    )


def test_good_map_passes():
    report = validate_arrays({"x": _smooth_sphere_map()})
    assert report.ok, report.summary()


def test_longitudinal_seam_detected():
    bad = _smooth_sphere_map()
    bad[:, : bad.shape[1] // 2] += 0.5  # hard step at the wrap and in the middle
    report = Report()
    check_wrap_continuity(bad, "bad", report)
    assert not report.ok


def test_pole_pinch_detected():
    bad = _smooth_sphere_map()
    rng = np.random.default_rng(0)
    bad[0, :] = rng.uniform(0.0, 1.0, bad.shape[1])  # noisy pole row
    report = Report()
    check_pole_rows(bad, "bad", report)
    assert not report.ok


def test_nan_detected():
    bad = _smooth_sphere_map()
    bad[10, 10] = np.nan
    report = validate_arrays({"bad": bad})
    assert not report.ok


def test_rgb_maps_supported():
    arr = np.stack([_smooth_sphere_map()] * 3, axis=-1)
    report = validate_arrays({"rgb": arr})
    assert report.ok, report.summary()

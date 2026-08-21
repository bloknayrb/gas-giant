from __future__ import annotations

import numpy as np
import pytest

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


# -- half-float EXR quantization ----------------------------------------------


def test_half_quantization_floor_tracks_the_representable_step():
    """float16's ulp crosses ABS_FLOOR at 2.0, so the floor must follow the data.

    Below 2.0 the flat-image floor still dominates and nothing changes; above it
    a single representable step is larger than ABS_FLOOR and would otherwise be
    reported as a real discontinuity."""
    from gasgiant.validate.seams import ABS_FLOOR, quantization_floor

    lo = np.full((4, 4), 1.5, dtype=np.float32)
    hi = np.full((4, 4), 2.5, dtype=np.float32)

    assert quantization_floor(lo, half=False) == ABS_FLOOR
    assert quantization_floor(hi, half=False) == ABS_FLOOR
    assert quantization_floor(lo, half=True) == ABS_FLOOR          # ulp 9.8e-4 < floor
    assert quantization_floor(hi, half=True) == pytest.approx(1.953e-3, rel=1e-3)


def test_half_quantization_floor_ignores_non_finite():
    from gasgiant.validate.seams import ABS_FLOOR, quantization_floor

    a = np.array([[1.0, np.inf, np.nan]], dtype=np.float32)
    assert quantization_floor(a, half=True) == ABS_FLOOR
    assert quantization_floor(np.array([], dtype=np.float32), half=True) == ABS_FLOOR


def test_one_ulp_seam_on_a_half_map_is_not_a_discontinuity():
    """The concrete failure the floor exists to prevent: a near-flat but nonzero
    half map (an aurora alpha channel) whose interior quantizes to exactly flat,
    with a single-ulp difference across the wrap seam. Without a
    quantization-aware floor the limit collapses to ABS_FLOOR and the check
    fails on rounding alone."""
    from gasgiant.validate.seams import Report, check_wrap_continuity, quantization_floor

    base = np.float16(2.5)
    a = np.full((32, 64), float(base), dtype=np.float32)
    a[:, 0] = float(np.nextafter(base, np.float16(100.0)))   # seam column, +1 ulp

    strict = Report()
    check_wrap_continuity(a, "alpha", strict)                 # old behaviour
    assert not strict.ok, "test is vacuous: a 1-ulp seam already passed"

    aware = Report()
    check_wrap_continuity(a, "alpha", aware, abs_floor=quantization_floor(a, half=True))
    assert aware.ok, "a single representable step must not read as a seam"

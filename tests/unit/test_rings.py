"""T16 Saturn rings: the radial ring strip (CPU, no GL).

The ring strip is a bounded optical-depth profile + seeded fine grain, so it is
fully unit-testable without a GPU. GL-dependent export round-trips (rings.exr
written / not written) live in tests/gpu/test_rings_export.py.
"""

from __future__ import annotations

import numpy as np

from gasgiant.export.rings import RING_HEIGHT, RING_WIDTH, ring_strip
from gasgiant.params.model import PlanetParams, RingsParams


def _enabled(seed: int = 7, **rings) -> PlanetParams:
    return PlanetParams(seed=seed, rings={"enabled": True, **rings})


def test_rings_disabled_by_default_no_op():
    """RingsParams default enabled=False -- a default planet exports no rings."""
    assert RingsParams().enabled is False
    assert PlanetParams().rings.enabled is False


def test_ring_strip_shape_and_bounds():
    strip = ring_strip(_enabled())
    assert strip.shape == (RING_WIDTH, RING_HEIGHT, 4)
    assert strip.dtype == np.float32
    assert strip.min() >= 0.0
    assert strip.max() <= 1.0
    assert np.isfinite(strip).all()


def test_ring_strip_deterministic():
    """Same seed + params -> byte-identical strip."""
    a = ring_strip(_enabled(seed=99))
    b = ring_strip(_enabled(seed=99))
    np.testing.assert_array_equal(a, b)


def test_ring_strip_seed_changes_grain():
    """Different master seed -> different fine-grain (the profile is the same
    table, but the seeded ringlet variation differs)."""
    a = ring_strip(_enabled(seed=1))
    b = ring_strip(_enabled(seed=2))
    assert not np.array_equal(a, b)


def test_cassini_division_is_a_gap_between_bright_bands():
    """Gross structure: the Cassini division (radius fraction ~0.70-0.76) is a
    low-opacity gap sitting between the dense B ring (inner) and the A ring
    (outer). Assert the mean alpha dips there well below both flanking bands."""
    alpha = ring_strip(_enabled(fine_grain=0.0))[..., 3].mean(axis=1)  # radial mean

    def band(lo: float, hi: float) -> float:
        return float(alpha[int(lo * RING_WIDTH) : int(hi * RING_WIDTH)].mean())

    b_ring = band(0.40, 0.60)      # dense B ring
    cassini = band(0.71, 0.76)     # the division (gap)
    a_ring = band(0.80, 0.92)      # A ring
    assert cassini < 0.5 * b_ring, (cassini, b_ring)
    assert cassini < 0.5 * a_ring, (cassini, a_ring)
    # Inner C-edge and outer A-edge fade to (near) transparent.
    assert alpha[0] < 0.1
    assert alpha[-1] < 0.1


def test_opacity_zero_is_fully_transparent():
    strip = ring_strip(_enabled(opacity=0.0))
    np.testing.assert_array_equal(strip[..., 3], 0.0)

"""GPU: storms.rim_contrast scales the hero perimeter ring + collar amplitude."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _hero_params(rim_contrast: float = 1.0) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    p.storms.rim_contrast = rim_contrast
    return p


def _t0(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]


def _row_band(t0: np.ndarray, lat_lo: float, lat_hi: float) -> np.ndarray:
    h = t0.shape[0]
    lats = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    return t0[(np.abs(lats) > lat_lo) & (np.abs(lats) < lat_hi)]


def test_rim_contrast_default_is_byte_identical(gpu):
    """rim_contrast=1.0 explicit vs default must produce bit-identical T0."""
    sim_default = Simulation(_hero_params(rim_contrast=1.0), gpu)
    sim_explicit = Simulation(_hero_params(rim_contrast=1.0), gpu)
    t0_default = _t0(sim_default)
    t0_explicit = _t0(sim_explicit)
    np.testing.assert_array_equal(t0_default, t0_explicit)


def test_rim_contrast_scales_hero_ring(gpu):
    """rim_contrast=2.0 vs 1.0 should differ near the hero but not far away."""
    sim_1x = Simulation(_hero_params(rim_contrast=1.0), gpu)
    sim_2x = Simulation(_hero_params(rim_contrast=2.0), gpu)
    t0_1x = _t0(sim_1x)
    t0_2x = _t0(sim_2x)

    diff = np.abs(t0_2x - t0_1x)

    # Overall: arrays must differ somewhere near the hero
    assert diff.max() > 1e-3, "rim_contrast=2.0 produced no change vs 1.0"

    # Hero latitude band: rows within ±10 deg of -22.5 deg
    band_lo = np.deg2rad(12.5)
    band_hi = np.deg2rad(32.5)
    hero_diff = _row_band(diff, band_lo, band_hi)
    hero_max = float(hero_diff.max())

    # Far-away band: around +50 deg — should be nearly untouched
    far_lo = np.deg2rad(40.0)
    far_hi = np.deg2rad(60.0)
    far_diff = _row_band(diff, far_lo, far_hi)
    far_max = float(far_diff.max())

    assert hero_max > 1e-3, "Expected visible change near hero latitude"
    assert far_max < 0.1 * hero_max, (
        f"Far-band change ({far_max:.5f}) should be < 10% of hero-band change "
        f"({hero_max:.5f}), but the ring/collar appears to affect the whole map"
    )


def test_rim_contrast_zero_removes_ring(gpu):
    """rim_contrast=0.0 should remove the dark ring + bright collar vs default."""
    sim_0 = Simulation(_hero_params(rim_contrast=0.0), gpu)
    sim_1 = Simulation(_hero_params(rim_contrast=1.0), gpu)
    t0_0 = _t0(sim_0)
    t0_1 = _t0(sim_1)

    diff = np.abs(t0_0 - t0_1)

    # The ring/collar should be gone at hero latitude
    hero_lo = np.deg2rad(12.5)
    hero_hi = np.deg2rad(32.5)
    hero_diff = _row_band(diff, hero_lo, hero_hi)
    assert float(hero_diff.max()) > 1e-3, (
        "rim_contrast=0.0 produced no change near hero; "
        "ring/collar was not removed"
    )

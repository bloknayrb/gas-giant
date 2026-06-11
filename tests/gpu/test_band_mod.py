"""GPU: the band-stamp modifiers actually shape the tracer field."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _quick(seed: int = 21, **bands) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 0  # keep T0 statistics band-driven
    for key, value in bands.items():
        setattr(p.bands, key, value)
    return p


def _t0_rows(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]


def _row_band(t0: np.ndarray, lat_lo: float, lat_hi: float) -> np.ndarray:
    h = t0.shape[0]
    lats = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    return t0[(np.abs(lats) > lat_lo) & (np.abs(lats) < lat_hi)]


def test_contrast_envelope_collapses_high_latitude_banding(gpu):
    base = Simulation(_quick(), gpu)
    faded = Simulation(_quick(contrast_envelope=1.0), gpu)
    # Sample where the envelope is strong (smoothstep(45, 72) > 0.75).
    lo, hi = np.deg2rad(63.0), np.deg2rad(72.0)
    spread_base = float(np.ptp(_row_band(_t0_rows(base), lo, hi).mean(axis=1)))
    spread_faded = float(np.ptp(_row_band(_t0_rows(faded), lo, hi).mean(axis=1)))
    assert spread_faded < 0.45 * spread_base
    # Equatorward banding is untouched.
    eq_base = _row_band(_t0_rows(base), 0.0, np.deg2rad(35.0)).mean(axis=1)
    eq_faded = _row_band(_t0_rows(faded), 0.0, np.deg2rad(35.0)).mean(axis=1)
    np.testing.assert_allclose(eq_base, eq_faded, atol=1e-5)


def test_variance_adds_longitudinal_drift(gpu):
    base = Simulation(_quick(), gpu)
    varied = Simulation(_quick(variance_amount=0.25), gpu)
    # Within-row (longitudinal) T0 std rises in the band region.
    lo, hi = 0.0, np.deg2rad(45.0)
    std_base = _row_band(_t0_rows(base), lo, hi).std(axis=1).mean()
    std_varied = _row_band(_t0_rows(varied), lo, hi).std(axis=1).mean()
    assert std_varied > std_base * 1.3


def test_faded_sector_lightens_its_longitude_window(gpu):
    p = _quick(faded_sector=1.0)
    sim = Simulation(p, gpu)
    lat_lo, lat_hi, lon, halfwidth = sim.profiles.fade_sector
    t0 = _t0_rows(sim)
    h, w = t0.shape
    lats = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    lons = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    rows = (lats > lat_lo) & (lats < lat_hi)
    dlon = np.abs(np.angle(np.exp(1j * (lons - lon))))
    inside = t0[np.ix_(rows, dlon < 0.4 * halfwidth)].mean()
    outside = t0[np.ix_(rows, dlon > 1.5 * halfwidth)].mean()
    assert inside > outside + 0.05  # the belt pales toward the mid tone

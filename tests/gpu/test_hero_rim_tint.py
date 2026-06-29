"""GPU tests for storms.hero_rim_tint (the dark reddish collar / Red Spot Hollow rim).

The hero's perimeter currently only DARKENS (a T0 dip); visual review found the spot
reads as a stain on the band with no distinct dark-reddish rim. hero_rim_tint adds a
reddening (T3 up) + extra darkening (T0 down) on the perimeter annulus (q~1.0-1.2) so
the oval reads as a discrete vortex with a dark-red collar.

Invariants:
  1. hero_rim_tint=0 takes a guarded branch => BYTE-IDENTICAL on T0 and T3, even with
     another hero lever (hero_rim_warp) on.
  2. hero_rim_tint>0 reddens (raises T3) in the hero band and stays localized.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(rim_tint: float | None = None, rim_warp: float = 0.0) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_rim_warp = rim_warp
    if rim_tint is not None:
        p.storms.hero_rim_tint = rim_tint
    return p


def _tracers(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


def _hero_band_rows(h: int, half_deg: float = 15.0) -> np.ndarray:
    lats = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lo = np.deg2rad(HERO_LAT_DEG - half_deg)
    hi = np.deg2rad(HERO_LAT_DEG + half_deg)
    return (lats >= lo) & (lats <= hi)


# ------------------------------------------------------------- byte-identity

def test_rim_tint_off_byte_identical_with_warp_on(gpu):
    base = _tracers(Simulation(_params(rim_warp=0.5), gpu))
    same = _tracers(Simulation(_params(rim_warp=0.5, rim_tint=0.0), gpu))
    np.testing.assert_array_equal(base[..., 0], same[..., 0])
    np.testing.assert_array_equal(base[..., 3], same[..., 3])


# ------------------------------------------------------------- effect + locality

def test_rim_tint_reddens_perimeter_and_stays_local(gpu):
    off = _tracers(Simulation(_params(rim_tint=0.0), gpu))
    on = _tracers(Simulation(_params(rim_tint=0.8), gpu))
    off_t3, on_t3 = off[..., 3], on[..., 3]

    # Reddens: T3 rises somewhere in the hero band.
    h = off_t3.shape[0]
    band = _hero_band_rows(h)
    assert (on_t3 - off_t3)[band].max() > 1e-3, "hero_rim_tint did not raise T3 (redden)"

    # Localized: far north (no hero) byte-identical on both channels.
    far = np.zeros(h, dtype=bool)
    far[: h // 4] = True
    assert not (band & far).any()
    np.testing.assert_array_equal(on_t3[far], off_t3[far])
    np.testing.assert_array_equal(on[..., 0][far], off[..., 0][far])

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


def _hero_q(shape: tuple[int, ...], hero) -> np.ndarray:
    """Normalized radius q = great-circle distance / r_core at every pixel, relative
    to the hero center. Valid for the default ROUND hero (aspect 1.0); same metric
    the shader uses for the rim/collar Gaussians."""
    h, w = shape[:2]
    lat = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lon = -np.pi + (np.arange(w) + 0.5) / w * 2.0 * np.pi
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    px, py, pz = np.cos(la) * np.cos(lo), np.sin(la), np.cos(la) * np.sin(lo)
    hx = np.cos(hero.lat) * np.cos(hero.lon)
    hy = np.sin(hero.lat)
    hz = np.cos(hero.lat) * np.sin(hero.lon)
    d = np.arccos(np.clip(px * hx + py * hy + pz * hz, -1.0, 1.0))
    return d / hero.r_core


# ------------------------------------------------------------- byte-identity

def test_rim_tint_off_byte_identical_with_warp_on(gpu):
    base = _tracers(Simulation(_params(rim_warp=0.5), gpu))
    same = _tracers(Simulation(_params(rim_warp=0.5, rim_tint=0.0), gpu))
    np.testing.assert_array_equal(base[..., 0], same[..., 0])
    np.testing.assert_array_equal(base[..., 3], same[..., 3])


# ------------------------------------------------------------- effect + locality

def test_rim_tint_reddens_perimeter_and_stays_local(gpu):
    off = _tracers(Simulation(_params(rim_tint=0.0), gpu))
    on_sim = Simulation(_params(rim_tint=0.8), gpu)
    on = _tracers(on_sim)
    off_t3, on_t3 = off[..., 3], on[..., 3]
    diff = on_t3 - off_t3

    # Reddening is concentrated on the PERIMETER ANNULUS (the Red Spot Hollow rim,
    # rring = exp(-(q-1.08)^2*11)), not the deep core: the base tint b.z*core is the
    # same on/off so it cancels in the diff, leaving only the rring contribution.
    h = off_t3.shape[0]
    hero = on_sim.vortices.heroes()[0]
    q = _hero_q(on_t3.shape, hero)
    annulus = (q >= 0.9) & (q <= 1.3)
    core = q < 0.6
    assert diff[annulus].max() > 1e-3, "hero_rim_tint did not redden the perimeter"
    assert diff[annulus].max() > 3.0 * np.abs(diff[core]).max(), (
        "rim_tint reddening is not concentrated at the perimeter annulus"
    )

    # Localized: far north (no hero) byte-identical on both channels.
    band = _hero_band_rows(h)
    far = np.zeros(h, dtype=bool)
    far[: h // 4] = True
    assert not (band & far).any()
    np.testing.assert_array_equal(on_t3[far], off_t3[far])
    np.testing.assert_array_equal(on[..., 0][far], off[..., 0][far])

"""GPU tests for storms.hero_rim_warp (the lumpy-oval boundary lever).

The hero's dark perimeter ring + bright collar are perfect azimuthally-symmetric
radial Gaussians in q (vortex_stamp.glsl) — a flawless oval edge that reads
"over-regular". hero_rim_warp warps the q feeding those two Gaussians with a
low-azimuthal-wavenumber (lobed) per-hero perturbation, so the boundary becomes
naturally irregular. Scale-invariant lobes (not pixel-frequency noise) so it holds
up at both full-disk and close-up.

Invariants:
  1. rim_warp=0.0 default takes a guarded (un-entered) branch in vortex_stamp.glsl,
     so an explicit 0.0 must be BYTE-IDENTICAL to the default render — even when
     ANOTHER hero lever (hero_mottle) is on, proving the guard fully short-circuits.
  2. rim_warp>0 must change the stamped field in the hero's rim/collar annulus, and
     the change must stay LOCALIZED to the hero (the far field is byte-identical;
     the downstream wake wedge uses along/across, not q, so it is untouched).
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(rim_warp: float | None = None, mottle: float = 0.0) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_mottle = mottle
    if rim_warp is not None:
        p.storms.hero_rim_warp = rim_warp
    return p


def _tracers(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


def _hero_band_rows(h: int, half_deg: float = 15.0) -> np.ndarray:
    lats = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lo = np.deg2rad(HERO_LAT_DEG - half_deg)
    hi = np.deg2rad(HERO_LAT_DEG + half_deg)
    return (lats >= lo) & (lats <= hi)


# ---------------------------------------------------------------- byte-identity

def test_default_rim_warp_byte_identical(gpu):
    """Explicit 0.0 must be byte-identical to default on T0 (the rim/collar channel)."""
    base = _tracers(Simulation(_params(), gpu))
    same = _tracers(Simulation(_params(rim_warp=0.0), gpu))
    np.testing.assert_array_equal(base[..., 0], same[..., 0])


def test_rim_warp_off_byte_identical_with_mottle_on(gpu):
    """rim_warp=0.0 with hero_mottle on must be byte-identical to mottle-on alone —
    the guard must remove the warp from the executed path regardless of co-params."""
    field_default = _tracers(Simulation(_params(mottle=0.35), gpu))[..., 0]
    field_explicit = _tracers(Simulation(_params(mottle=0.35, rim_warp=0.0), gpu))[..., 0]
    np.testing.assert_array_equal(field_default, field_explicit)


# ---------------------------------------------------------------- effect + locality

def test_rim_warp_changes_boundary_and_stays_local(gpu):
    off = _tracers(Simulation(_params(rim_warp=0.0), gpu))[..., 0]
    on = _tracers(Simulation(_params(rim_warp=0.6), gpu))[..., 0]

    # Effect exists.
    assert np.abs(on - off).max() > 1e-3, "hero_rim_warp produced no visible change"

    # Localized: the far north (no hero there, and no wake) is byte-identical.
    h = off.shape[0]
    far = np.zeros(h, dtype=bool)
    far[: h // 4] = True
    assert not (_hero_band_rows(h) & far).any()
    np.testing.assert_array_equal(on[far], off[far])

    # The change is structured (a warped boundary), not a flat shift, inside the
    # hero band.
    band = _hero_band_rows(h)
    diff_band = np.where(band[:, None], on - off, 0.0)
    assert diff_band.std() > 1e-4, "rim_warp added no spatial structure to the boundary"

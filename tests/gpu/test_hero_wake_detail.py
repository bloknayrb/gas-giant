"""GPU tests for storms.hero_wake_detail (filament structure in the hero wake).

The hero's downstream wake is stamped as a smooth exp(-across^2) wedge into the
relaxation TARGET (vortex_stamp.glsl), so even though the wake VELOCITY is turbulent
(psi.comp's u_wake_gain), relaxation keeps pulling the brightness back to a smooth blob.
hero_wake_detail breaks that target into ragged, intermittent, flow-aligned filaments:
it perturbs the wedge envelope and carves the interior with an anisotropic fbm.

Invariants:
  1. hero_wake_detail=0 takes a guarded branch => BYTE-IDENTICAL on T0 and T3, even with
     other hero levers (hero_rim_tint / hero_rim_warp) on.
  2. hero_wake_detail>0 changes the DOWNSTREAM wake (T0) while leaving the upstream
     hemisphere byte-identical (the wake block only touches along>0).
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(
    wake_detail: float | None = None,
    rim_tint: float = 0.0,
    rim_warp: float = 0.0,
) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_rim_tint = rim_tint
    p.storms.hero_rim_warp = rim_warp
    if wake_detail is not None:
        p.storms.hero_wake_detail = wake_detail
    return p


def _tracers(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


def _lon_of_columns(w: int) -> np.ndarray:
    """Longitude (radians) at each equirect column center, matching the shader's
    atan2(p.z, p.x): col c -> lon = -pi + (c+0.5)/w * 2pi."""
    return -np.pi + (np.arange(w) + 0.5) / w * 2.0 * np.pi


# ------------------------------------------------------------- byte-identity

def test_wake_detail_off_byte_identical_with_other_levers_on(gpu):
    """wake_detail=0 is a guarded no-op: byte-identical even with rim_tint + rim_warp on."""
    base = _tracers(Simulation(_params(rim_tint=0.7, rim_warp=0.5), gpu))
    same = _tracers(Simulation(_params(wake_detail=0.0, rim_tint=0.7, rim_warp=0.5), gpu))
    np.testing.assert_array_equal(base[..., 0], same[..., 0])
    np.testing.assert_array_equal(base[..., 3], same[..., 3])


# ------------------------------------------------------------- effect + locality

def test_wake_detail_textures_downstream_wake(gpu):
    """wake_detail>0 changes the downstream wake brightness; upstream hemisphere is
    untouched (the wake block only stamps along>0, downstream of the hero)."""
    off = _tracers(Simulation(_params(wake_detail=0.0), gpu))
    on_sim = Simulation(_params(wake_detail=0.8), gpu)
    on = _tracers(on_sim)
    off_t0, on_t0 = off[..., 0], on[..., 0]

    # The wake is downstream of the hero. Regardless of flow direction, SOME column
    # region changes by more than the (zero) kinematic noise floor.
    assert np.abs(on_t0 - off_t0).max() > 1e-3, "hero_wake_detail did not change the wake"

    # Upstream byte-identity: the wake block only stamps along>0 (downstream in
    # longitude; along = dlon * wake_dir, matching vortex_stamp.glsl's
    # dlon = mod(plon - vlon + 3*PI, 2*PI) - PI and along = dlon * down). So no pixel
    # on the UPSTREAM longitude side may change. (1e-6 floor: taking the guarded fbm
    # branch shifts denormal flush ~1e-23, ~20 orders below the 0.16-scale wake.)
    hero = on_sim.vortices.heroes()[0]
    lon = _lon_of_columns(on_t0.shape[1])
    dlon = np.mod(lon - hero.lon + 3.0 * np.pi, 2.0 * np.pi) - np.pi
    along = dlon * hero.wake_dir                       # >0 downstream, <0 upstream
    changed = np.abs(on_t0 - off_t0) > 1e-6            # (h, w)
    upstream = along < -1e-3                           # (w,) columns upstream of hero
    assert not changed[:, upstream].any(), "hero_wake_detail changed the upstream side"

    # Locality: the far-north quarter (no hero, no wake) is unchanged. Use a tiny
    # floor rather than exact equality: taking the guarded fbm branch shifts
    # denormal flush-to-zero for the near-zero T3 values up there (~1e-23), which is
    # ~20 orders of magnitude below any real wake signal (0.16-scale). The hero is in
    # the south, so a real leak would be obvious; this proves none.
    h = off_t0.shape[0]
    far = np.zeros(h, dtype=bool)
    far[: h // 4] = True
    assert np.abs(on_t0[far] - off_t0[far]).max() < 1e-6
    assert np.abs(on[..., 3][far] - off[..., 3][far]).max() < 1e-6

"""GPU tests for detail.hero_calm (calm the band-aligned grain in the hero interior).

The detail-synthesis filament streak + striation are band/flow-aligned and are
AMPLIFIED near heroes (the 1+1.4*hero factor), so they cross the GRS as straight
"wood-grain" that ignores the vortex rotation (visual-review finding). hero_calm
attenuates those two band-aligned terms inside the hero (weighted by the heroMask),
so the vortex-aligned hero_spiral lanes + the sim-side hero_mottle churn carry the
interior instead.

Invariants:
  1. hero_calm=0 multiplies the band terms by exactly 1.0 => BYTE-IDENTICAL to not
     setting it, even with striations on (the calm factor is `1 - calm*hero`, and
     `w*1.0 == w` in IEEE).
  2. hero_calm>0 REDUCES the detail-field texture energy INSIDE the hero (lower local
     std) and stays LOCALIZED (the far field is byte-identical).
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.engine.snapshot import hero_centers
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(hero_calm: float | None = None) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = HERO_LAT_DEG
    p.detail.striation_amount = 1.0   # band-aligned grain to calm
    if hero_calm is not None:
        p.detail.hero_calm = hero_calm
    return p


def _synth(params: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(params, gpu)
    s = sim.solver
    out = gpu.texture2d((512, 256), 1, "f4", linear=True)
    sim.detail_synth.synthesize(
        params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
        sim.profile_dyn, out, params.detail,
        heroes=hero_centers(sim.vortices),
    )
    field = gpu.read_texture(out)[..., 0]
    out.release()
    return field


def _hero_band_rows(h: int, half_deg: float = 15.0) -> np.ndarray:
    lats = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lo = np.deg2rad(HERO_LAT_DEG - half_deg)
    hi = np.deg2rad(HERO_LAT_DEG + half_deg)
    return (lats >= lo) & (lats <= hi)


def _hero_box(off: np.ndarray, on: np.ndarray, half: int = 18) -> tuple[slice, slice]:
    h, w = off.shape
    band = _hero_band_rows(h)
    diff = np.where(band[:, None], np.abs(on - off), 0.0)
    r, c = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return (slice(max(r - half, 0), min(r + half, h)),
            slice(max(c - half, 0), min(c + half, w)))


# ------------------------------------------------------------- byte-identity

def test_hero_calm_off_byte_identical(gpu):
    """hero_calm=0 (with striations on) must be byte-identical to not setting it."""
    field_default = _synth(_params(), gpu)
    field_explicit = _synth(_params(hero_calm=0.0), gpu)
    np.testing.assert_array_equal(field_default, field_explicit)


# ------------------------------------------------------------- effect + locality

def test_hero_calm_reduces_interior_grain_and_stays_local(gpu):
    off = _synth(_params(hero_calm=0.0), gpu)
    on = _synth(_params(hero_calm=0.9), gpu)

    # Effect exists.
    assert np.abs(on - off).max() > 1e-3, "hero_calm produced no visible change"

    # Localized: far north (no hero) is byte-identical.
    h = off.shape[0]
    far = np.zeros(h, dtype=bool)
    far[: h // 4] = True
    assert not (_hero_band_rows(h) & far).any()
    np.testing.assert_array_equal(on[far], off[far])

    # It CALMS: the interior texture energy (local std) drops.
    box = _hero_box(off, on)
    assert on[box].std() < off[box].std(), (
        f"hero_calm did not reduce interior texture (off.std={off[box].std():.4f}, "
        f"on.std={on[box].std():.4f})"
    )

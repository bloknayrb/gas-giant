"""GPU tests for storms.hero_mottle and storms.hero_tint_var.

Two invariants:
  1. Both default to 0.0 and take a guarded (un-entered) branch in
     vortex_stamp.glsl, so an explicit 0.0 must be BYTE-IDENTICAL to the
     default render on both the T0 (brightness) and T3 (tint) channels.
  2. With a lever on, the change must be LOCALIZED to the hero (the far field
     stays byte-identical) and must add interior structure (raise local
     variance) inside the hero core.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(mottle: float = 0.0, tint: float = 0.0) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_mottle = mottle
    p.storms.hero_tint_var = tint
    return p


def _tracers(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


def _hero_band_rows(h: int, half_deg: float = 15.0) -> np.ndarray:
    lats = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lo = np.deg2rad(HERO_LAT_DEG - half_deg)
    hi = np.deg2rad(HERO_LAT_DEG + half_deg)
    return (lats >= lo) & (lats <= hi)


def _hero_box(off: np.ndarray, on: np.ndarray, half: int = 18) -> tuple[slice, slice]:
    """Locate the hero by the peak-|diff| cell inside its latitude band — the
    only thing the lever changes — and return a crop box around it."""
    h, w = off.shape
    band = _hero_band_rows(h)
    diff = np.abs(on - off)
    diff_band = np.where(band[:, None], diff, 0.0)
    r, c = np.unravel_index(int(np.argmax(diff_band)), diff_band.shape)
    return (slice(max(r - half, 0), min(r + half, h)),
            slice(max(c - half, 0), min(c + half, w)))


# ---------------------------------------------------------------- byte-identity

def test_default_mottle_tint_byte_identical(gpu):
    """Explicit (0.0, 0.0) must be byte-identical to default on T0 and T3."""
    base = _tracers(Simulation(_params(), gpu))
    same = _tracers(Simulation(_params(mottle=0.0, tint=0.0), gpu))
    np.testing.assert_array_equal(base[..., 0], same[..., 0])
    np.testing.assert_array_equal(base[..., 3], same[..., 3])


# ---------------------------------------------------------------- mottle effect (T0)

def test_mottle_breaks_up_core_and_stays_local(gpu):
    off = _tracers(Simulation(_params(), gpu))[..., 0]
    on = _tracers(Simulation(_params(mottle=0.8), gpu))[..., 0]

    # Effect exists.
    assert np.abs(on - off).max() > 1e-3, "hero_mottle produced no visible change"

    # Localized: the northern hemisphere (no hero there) is byte-identical.
    h = off.shape[0]
    north = _hero_band_rows(h)  # hero band (south)
    far = np.zeros(h, dtype=bool)
    far[: h // 4] = True  # far north rows
    assert not (north & far).any()
    np.testing.assert_array_equal(on[far], off[far])

    # Interior structure: the ADDED field (on-off) varies spatially inside the
    # core — i.e. mottle, not a flat brightness shift. (A raw std(on) vs std(off)
    # comparison is swamped by the hero's own rim/collar gradient.)
    box = _hero_box(off, on)
    assert (on - off)[box].std() > 1e-3, (
        f"mottle added no spatial structure to T0 (diff_box.std="
        f"{(on - off)[box].std():.5f})"
    )


# ---------------------------------------------------------------- tint effect (T3)

def test_tint_var_modulates_core_tint(gpu):
    off = _tracers(Simulation(_params(), gpu))[..., 3]
    on = _tracers(Simulation(_params(tint=0.8), gpu))[..., 3]

    assert np.abs(on - off).max() > 1e-3, "hero_tint_var produced no visible change"

    # The added tint field varies spatially inside the core (festoon, not a flat
    # tint shift).
    box = _hero_box(off, on)
    assert (on - off)[box].std() > 1e-3, (
        f"tint_var added no spatial structure to T3 (diff_box.std="
        f"{(on - off)[box].std():.5f})"
    )

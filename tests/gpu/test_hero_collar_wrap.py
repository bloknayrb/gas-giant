"""GPU tests for detail.hero_collar_wrap (the GRS 'hollow' collar lanes).

Invariants:
  1. collar_wrap=0 must be BYTE-IDENTICAL to not setting it, even when ANOTHER
     DETAIL_FX param is on (hero_spiral=0.7) — the runtime guard removes the term
     from the executed path, so it cannot perturb the existing spiral output.
     This is the critical short-circuit assert: collar-wrap shares the FX program
     with hero_spiral, so a 0-amplitude must change nothing.
  2. collar_wrap>0 must change the detail field in the collar annulus, and it
     must work INDEPENDENTLY of hero_spiral (hero_spiral=0).
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.engine.snapshot import hero_centers
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _params(hero_spiral: float = 0.0, collar_wrap: float | None = None) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    p.detail.hero_spiral = hero_spiral
    if collar_wrap is not None:
        p.detail.hero_collar_wrap = collar_wrap
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


# ------------------------------------------------- byte-identity with FX on

def test_collar_wrap_off_byte_identical_with_spiral_on(gpu):
    """hero_spiral=0.7 with collar_wrap=0 explicit must be byte-identical to
    hero_spiral=0.7 with collar_wrap unset — the FX-on / wrap-off short-circuit."""
    field_default = _synth(_params(hero_spiral=0.7), gpu)
    field_explicit = _synth(_params(hero_spiral=0.7, collar_wrap=0.0), gpu)
    np.testing.assert_array_equal(field_default, field_explicit)


# ------------------------------------------------- effect, independent of spiral

def test_collar_wrap_adds_structure_without_spiral(gpu):
    """collar_wrap=0.8 with hero_spiral=0 must change the detail field — proving
    it is independent of hero_spiral and that fx_on triggers on collar_wrap alone."""
    off = _synth(_params(hero_spiral=0.0, collar_wrap=0.0), gpu)
    on = _synth(_params(hero_spiral=0.0, collar_wrap=0.8), gpu)
    assert np.abs(on - off).max() > 1e-3, (
        "collar_wrap produced no change with hero_spiral=0 — not independent"
    )

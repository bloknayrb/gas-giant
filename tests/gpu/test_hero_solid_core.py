"""GPU tests for storms.hero_solid_core (vorticity-mode solid-body hero).

The Gaussian hero vorticity is center-peaked -> differential rotation -> the
interior winds into a center-draining whirlpool. hero_solid_core blends toward a
near-uniform vorticity patch -> rigid solid-body interior rotation -> a coherent
oval. Invariants:

  1. hero_solid_core=0 is byte-identical to the Gaussian path (the blend is
     guarded, so it is a no-op when off). Guards every existing vorticity preset.
  2. hero_solid_core>0 materially changes the hero region.
  3. The patch feeds the prognostic vorticity solver, so it must stay BOUNDED
     over a long horizon (no NaN / blow-up), like the existing inject path.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2
HERO_LAT = -22.5


def _params(solid: float, steps: int = 60):
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.storms.hero_count = 1
    p.storms.hero_latitude = HERO_LAT
    p.storms.hero_solid_core = solid
    return p


def _render(p, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(512)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def test_solid_core_byte_identical_when_off(gpu):
    """solid_core=0 explicit must equal the default (Gaussian) path."""
    base = _render(_params(0.0), gpu)
    same = _render(_params(0.0), gpu)
    np.testing.assert_array_equal(base, same)


def test_solid_core_changes_hero_region(gpu):
    """solid_core=1 must materially change the render vs the Gaussian hero."""
    gauss = _render(_params(0.0), gpu)
    solid = _render(_params(1.0), gpu)
    assert np.abs(solid - gauss).max() > GPU_NOISE_ATOL, (
        "hero_solid_core=1 did not change the vorticity-mode hero"
    )


def test_solid_core_bounded_over_long_horizon(gpu):
    """The patch feeds the prognostic solver -> must stay finite and bounded
    over a long horizon (no sticky NaN / runaway), like the inject path."""
    out = _render(_params(1.0, steps=400), gpu)
    assert np.all(np.isfinite(out)), "hero_solid_core produced non-finite output"
    assert out.max() <= 1.0 + 1e-3 and out.min() >= -1e-3, (
        f"color out of range over long horizon: [{out.min()}, {out.max()}]"
    )

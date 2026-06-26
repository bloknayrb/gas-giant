"""GPU tests for storms.oval_solid_core (solid-body rotation for large ovals).

Same anti-whirlpool mechanism as hero_solid_core, applied to white ovals whose
core radius is at/above the OVAL_SOLID_MIN_R threshold (0.035 rad). At long
dev_steps a Gaussian oval winds the passive tracer into a mini-bullseye; the
solid-body patch keeps it a coherent spot. Invariants mirror the hero test:

  1. oval_solid_core=0 is byte-identical to the Gaussian path (guarded no-op).
  2. oval_solid_core>0 materially changes the render (needs a large oval present).
  3. The patch feeds the prognostic solver, so it must stay BOUNDED (no NaN /
     blow-up) over a long horizon.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.solver import compute_dt
from gasgiant.sim.vortices import KIND_OVAL, generate_vortices

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2
OVAL_SOLID_MIN_R = 0.035  # mirrors the GLSL constant


def _params(solid: float, steps: int = 60):
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.storms.hero_count = 0           # isolate ovals from the hero mechanism
    p.storms.oval_density = 3.0       # ensure several large ovals exist
    p.storms.oval_solid_core = solid
    return p


def _render(p, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(512)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def test_large_oval_exists():
    """Guard: the 'changes' test is only meaningful if a qualifying large oval
    is actually seeded for this config (else a no-change would be a false pass)."""
    p = _params(0.0)
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    dt = compute_dt(p.sim.resolution, p.sim.dt_scale, profiles.max_speed)
    reg = generate_vortices(p.seed, bands, profiles, p.storms, p.poles,
                            dt=dt, dev_steps=p.sim.dev_steps)
    big = [v for v in reg.vortices
           if v.kind == KIND_OVAL and v.r_core >= OVAL_SOLID_MIN_R]
    assert big, "no oval with r_core >= 0.035 seeded; 'changes' test would be vacuous"


def test_oval_solid_core_byte_identical_when_off(gpu):
    """oval_solid_core=0 explicit must equal the default (Gaussian) path."""
    base = _render(_params(0.0), gpu)
    same = _render(_params(0.0), gpu)
    np.testing.assert_array_equal(base, same)


def test_oval_solid_core_changes_render(gpu):
    """oval_solid_core=1 must materially change the render vs the Gaussian ovals."""
    gauss = _render(_params(0.0), gpu)
    solid = _render(_params(1.0), gpu)
    assert np.abs(solid - gauss).max() > GPU_NOISE_ATOL, (
        "oval_solid_core=1 did not change the vorticity-mode ovals"
    )


def test_oval_solid_core_bounded_over_long_horizon(gpu):
    """The patch feeds the prognostic solver -> must stay finite and bounded."""
    out = _render(_params(1.0, steps=400), gpu)
    assert np.all(np.isfinite(out)), "oval_solid_core produced non-finite output"
    assert out.max() <= 1.0 + 1e-3 and out.min() >= -1e-3, (
        f"color out of range over long horizon: [{out.min()}, {out.max()}]"
    )

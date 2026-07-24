"""GPU tests for M2 per-storm CastLevers — the CAST_LEVERS variant + binding-5 SSBO.

These are the runtime gates the p05 render-hash gate and the kinematic source-pin
cannot see (p05 renders only the DEFAULT program; the source-pin guards file text):

  1. the variant COMPILES and RUNS (a cast hero carrying an override renders at all);
  2. forced-variant no-op — a cast hero whose levers all equal the resolved GLOBAL
     values renders byte-identical to the default program, so the compiled variant is
     numerically inert when nothing truly overrides (the analogue of the emergence
     far-field no-op, but here proven by matching values rather than locality);
  3. per-storm-ness — SWAPPING which of two heroes carries an override changes the
     output, which is impossible if the value were still a single global uniform;
  4. dev-0 omega: a per-storm solid_core lands on the right hero's vorticity patch
     (the byte-exact vorticity carve-out — read before any SOR/advection).

Stamp levers (rim_*/mottle/tint_var/wake_detail) are exercised on the KINEMATIC path
(byte-exact developed tracers); solid_core on the vorticity omega_init texture.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import CastKind, PlanetParams, SolverType, StormOverride

pytestmark = pytest.mark.gpu


def _kin_params() -> PlanetParams:
    """Kinematic scene with every seeded population off, so only cast storms exist
    and developed tracers are byte-exact-comparable across full rebuilds."""
    p = PlanetParams(seed=17)
    p.sim.resolution = 512
    p.sim.dev_steps = 20
    p.storms.hero_count = 0
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.accent_count = 0
    p.storms.small_density = 0.0
    p.storms.outbreak_count = 0
    assert p.solver.type == SolverType.KINEMATIC
    return p


def _hero(lat, lon, **levers) -> StormOverride:
    return StormOverride(kind=CastKind.HERO, lat_deg=lat, lon_deg=lon, radius=0.1,
                         **levers)


def _tracers(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


# ----------------------------------------------------------- forced-variant no-op

def test_cast_levers_matching_global_is_byte_identical(gpu):
    """A cast hero whose 6 stamp levers all equal the GLOBAL values selects the
    CAST_LEVERS variant (values are non-None) yet must render byte-identical to the
    same hero with them left None (the default program). Proves the variant both
    COMPILES/RUNS and is numerically inert when it overrides nothing."""
    p_none = _kin_params()
    p_none.storms.cast = [_hero(-20.0, 0.0)]

    g = p_none.storms
    p_match = _kin_params()
    p_match.storms.cast = [_hero(
        -20.0, 0.0,
        rim_contrast=g.rim_contrast, rim_tint=g.hero_rim_tint,
        rim_warp=g.hero_rim_warp, mottle=g.hero_mottle,
        tint_var=g.hero_tint_var, wake_detail=g.hero_wake_detail,
    )]

    np.testing.assert_array_equal(_tracers(p_none, gpu), _tracers(p_match, gpu))


def test_cast_lever_override_changes_output(gpu):
    """A mottle override actually reaches the GPU: churning the interior changes the
    developed tracers vs the inherit-global (mottle 0) hero."""
    p_off = _kin_params()
    p_off.storms.cast = [_hero(-20.0, 0.0)]
    p_on = _kin_params()
    p_on.storms.cast = [_hero(-20.0, 0.0, mottle=0.9)]

    assert np.abs(_tracers(p_on, gpu).astype(np.float64)
                  - _tracers(p_off, gpu).astype(np.float64)).max() > 1e-3


def test_cast_levers_are_per_storm(gpu):
    """The proof M2 exists for: with TWO cast heroes, moving a mottle override from
    one to the other changes the output. If mottle were still a global uniform both
    scenes would be byte-identical (same global value, same two heroes)."""
    def scene(m_first: float, m_second: float) -> np.ndarray:
        p = _kin_params()
        p.storms.cast = [
            _hero(-20.0, -40.0, mottle=m_first),
            _hero(20.0, 40.0, mottle=m_second),
        ]
        return _tracers(p, gpu).astype(np.float64)

    a = scene(0.9, 0.0)   # churn on the southern hero
    b = scene(0.0, 0.9)   # churn on the northern hero
    assert np.abs(a - b).max() > 1e-3


# ----------------------------------------------------- dev-0 omega (solid_core)

def _vort_params(solid_core, second_solid_core) -> PlanetParams:
    """Vorticity scene, dev_steps 0, TWO cast heroes (so every comparison holds the
    background fixed and differs only in the solid_core levers); omega_init only."""
    p = PlanetParams(seed=17)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.solver.type = SolverType.VORTICITY
    p.storms.hero_count = 0
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.accent_count = 0
    p.storms.small_density = 0.0
    p.storms.outbreak_count = 0
    p.storms.cast = [
        _hero(-20.0, -40.0, solid_core=solid_core),
        _hero(20.0, 40.0, solid_core=second_solid_core),
    ]
    assert p.solver.type == SolverType.VORTICITY
    return p


def _dev0_omega(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    q = np.asarray(sim.gpu.read_texture(sim.solver._omega_state.cur))
    return np.squeeze(q).astype(np.float64)


def test_cast_solid_core_matching_global_is_byte_identical(gpu):
    """solid_core override equal to the global renders a byte-identical dev-0 omega
    (the vorticity byte-exact carve-out): the compiled CAST_LEVERS variant is inert
    on the omega path when it overrides nothing."""
    g = PlanetParams().storms.hero_solid_core
    p_none = _vort_params(solid_core=None, second_solid_core=None)
    p_match = _vort_params(solid_core=g, second_solid_core=g)
    np.testing.assert_array_equal(_dev0_omega(p_none, gpu), _dev0_omega(p_match, gpu))


def test_cast_solid_core_is_per_storm_on_omega(gpu):
    """Per-storm solid_core on the vorticity path: swapping which hero carries the
    solid-core patch changes the dev-0 omega texture."""
    a = _dev0_omega(_vort_params(solid_core=0.9, second_solid_core=0.0), gpu)
    b = _dev0_omega(_vort_params(solid_core=0.0, second_solid_core=0.9), gpu)
    assert np.abs(a - b).max() > 1e-3

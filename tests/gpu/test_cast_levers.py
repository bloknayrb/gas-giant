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


def test_cast_wake_detail_override_reaches_gpu(gpu):
    """wake_detail rides a SEPARATE index in the wake block (cast_lever_data[2*i+1].y,
    hand-written apart from the interior cl0/cl1 reads). Drive it non-zero — no other
    test exercises that column, so a typo there would otherwise pass silently."""
    p_off = _kin_params()
    p_off.storms.cast = [_hero(-20.0, 0.0)]
    p_on = _kin_params()
    p_on.storms.cast = [_hero(-20.0, 0.0, wake_detail=0.9)]
    assert np.abs(_tracers(p_on, gpu).astype(np.float64)
                  - _tracers(p_off, gpu).astype(np.float64)).max() > 1e-3


def test_cast_zero_default_columns_reach_gpu(gpu):
    """rim_tint (cl0.y), rim_warp (cl0.z), tint_var (cl1.x) all default to 0, so the
    forced-variant no-op cannot discriminate their columns. Drive them together and
    confirm the reads land (output differs from the all-global hero)."""
    p_off = _kin_params()
    p_off.storms.cast = [_hero(-20.0, 0.0)]
    p_on = _kin_params()
    p_on.storms.cast = [_hero(-20.0, 0.0, rim_tint=0.8, rim_warp=0.8, tint_var=0.8)]
    assert np.abs(_tracers(p_on, gpu).astype(np.float64)
                  - _tracers(p_off, gpu).astype(np.float64)).max() > 1e-3


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


# =============================================== M2-B: the emergence family
# emergence/shape/taper live in vec4_2 of the row and, unlike the M2-A levers,
# feed a CROSS-HERO max() combine (heroRelaxWeight) that used to scale by one
# global after its loop. The equivalence test below is the gate on that
# restructure; the swap tests prove the values are genuinely per storm.

def _emergent_kin(global_emergence: float = 0.9, **globals_) -> PlanetParams:
    """Kinematic cast-only scene running the emergence pack from the GLOBAL lever
    (hero_count 0, so the variant is selected by the cast hero's existence)."""
    p = _kin_params()
    p.storms.hero_emergence = global_emergence
    for k, v in globals_.items():
        setattr(p.storms, k, v)
    return p


def test_emergence_family_matching_global_is_byte_identical(gpu):
    """THE M2-B equivalence gate. TWO cast heroes at one global emergence, once
    with the levers left None (default program, emergence scaled once after
    heroRelaxWeight's cross-hero max) and once with every hero carrying that same
    value explicitly (CAST_LEVERS variant, each accumulator scaled by its owner's
    emergence). The restructure is only legitimate if these are byte-identical:
    max(E*a, E*b) picks the same candidate as E*max(a,b), and the final combine
    keeps the winner's emergence, so a uniform-emergence scene is unchanged.

    Two heroes is load-bearing — with one hero the max() never contends and the
    restructure would be trivially safe."""
    p_none = _emergent_kin(hero_shape=1.0, hero_taper=0.8)
    p_none.storms.cast = [_hero(-20.0, -40.0), _hero(20.0, 40.0)]

    p_match = _emergent_kin(hero_shape=1.0, hero_taper=0.8)
    p_match.storms.cast = [
        _hero(-20.0, -40.0, emergence=0.9, shape=1.0, taper=0.8),
        _hero(20.0, 40.0, emergence=0.9, shape=1.0, taper=0.8),
    ]

    np.testing.assert_array_equal(_tracers(p_none, gpu), _tracers(p_match, gpu))


def test_emergence_is_per_storm(gpu):
    """Swapping WHICH of two heroes is emergent changes the output — impossible if
    emergence were still one global uniform (both scenes share the same global, the
    same two hero positions and the same set of override values)."""
    def scene(first: float, second: float) -> np.ndarray:
        p = _emergent_kin()
        p.storms.cast = [
            _hero(-20.0, -40.0, emergence=first),
            _hero(20.0, 40.0, emergence=second),
        ]
        return _tracers(p, gpu).astype(np.float64)

    assert np.abs(scene(0.9, 0.0) - scene(0.0, 0.9)).max() > 1e-3


def test_cast_only_emergence_selects_the_variant(gpu):
    """Predicate gate: with the GLOBAL emergence at 0, a cast hero's own
    emergence override must still select HERO_EMERGENCE (dual-gated with
    CAST_LEVERS, which its own presence satisfies). Off-by-predicate would
    silently render the legacy stamped hero, so compare against emergence 0."""
    p_off = _emergent_kin(global_emergence=0.0)
    p_off.storms.cast = [_hero(-20.0, 0.0, emergence=0.0)]
    p_on = _emergent_kin(global_emergence=0.0)
    p_on.storms.cast = [_hero(-20.0, 0.0, emergence=0.9)]

    assert np.abs(_tracers(p_on, gpu).astype(np.float64)
                  - _tracers(p_off, gpu).astype(np.float64)).max() > 1e-3


def test_shape_and_taper_are_per_storm(gpu):
    """shape (vec4_2.y) and taper (vec4_2.z) ride their own columns and their own
    read sites (stamp anatomy, heroRelaxWeight's Rr/twr, heroBandDeflect's hold).
    Swap them between two equally-emergent heroes: a wrong column or a stale
    global would leave the two scenes identical."""
    def scene(first: tuple[float, float], second: tuple[float, float]) -> np.ndarray:
        p = _emergent_kin()
        p.storms.cast = [
            _hero(-20.0, -40.0, emergence=0.9, shape=first[0], taper=first[1]),
            _hero(20.0, 40.0, emergence=0.9, shape=second[0], taper=second[1]),
        ]
        return _tracers(p, gpu).astype(np.float64)

    assert np.abs(scene((1.4, 1.2), (0.0, 0.0))
                  - scene((0.0, 0.0), (1.4, 1.2))).max() > 1e-3


def test_per_storm_emergence_on_omega_needs_that_heros_solid_core(gpu):
    """Documented coupling, both directions, on the byte-exact dev-0 omega:

    (a) with solid_core 0 on both heroes the omega side never enters the ring
        branch, so per-storm emergence cannot move omega -> byte-identical;
    (b) with solid_core on, swapping which hero is emergent DOES move it."""
    def scene(solid: float, e_first: float, e_second: float) -> np.ndarray:
        p = _vort_params(solid_core=solid, second_solid_core=solid)
        p.storms.hero_emergence = 0.9
        p.storms.cast[0].emergence = e_first
        p.storms.cast[1].emergence = e_second
        return _dev0_omega(p, gpu)

    np.testing.assert_array_equal(scene(0.0, 0.9, 0.0), scene(0.0, 0.0, 0.9))
    assert np.abs(scene(0.9, 0.9, 0.0) - scene(0.9, 0.0, 0.9)).max() > 1e-3

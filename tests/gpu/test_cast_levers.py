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
     (the byte-exact vorticity carve-out — read before any SOR/advection);
  5. (M2-B) the emergence family — emergence/shape/taper — on both the stamp and
     the omega side, plus the two-spellings equivalence that keeps a cast-only
     emergent hero identical to the same storm expressed through the global.

Stamp levers (rim_*/mottle/tint_var/wake_detail) and the emergence family are
exercised on the KINEMATIC path (byte-exact developed tracers); solid_core and the
omega-side shape/taper on the vorticity omega_init texture.
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
    """wake_detail rides a SEPARATE index in the wake block (cast_lever_data[3*i+1].y,
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
# global after its loop. Each accumulator there now carries the emergence of the
# hero that owns it. NOTE what the equivalence test below does and does NOT gate:
# the `_e` companions are in the plain HERO_EMERGENCE program, NOT behind
# CAST_LEVERS, so BOTH sides of it run the restructured combine — it gates
# SSBO-vs-uniform value resolution. The restructure against the pre-M2-B
# single-scale arithmetic is gated by `scripts/m2b_emergence_hash.py --check`
# (machine-local baseline, kinematic/byte-exact), not by anything in-repo.

# Two heroes ~14 deg apart at one latitude: close enough that each sits inside
# the other's q<=4.2 rim/flush window AND its wake wedge, so heroRelaxWeight's
# cross-hero max() genuinely CONTENDS. (The obvious far-apart pair -- opposite
# hemispheres, 80+ deg -- has disjoint windows and silently degenerates into two
# independent single-hero scenes, where the max never contends at all.)
_PAIR = ((-20.0, -7.0), (-20.0, 7.0))


def _emergent_kin(emergence: float = 0.9, taper: float = 0.0) -> PlanetParams:
    """Kinematic cast-only scene running the emergence pack from the GLOBAL levers
    (hero_count 0, so the variant is selected by the cast heroes' existence).
    hero_shape rides its 1.0 model default — the outline deformation is already
    live in every scene here."""
    p = _kin_params()
    p.storms.hero_emergence = emergence
    p.storms.hero_taper = taper
    return p


def test_emergence_family_matching_global_is_byte_identical(gpu):
    """Forced-variant no-op for the emergence family: TWO CONTENDING cast heroes
    at one global emergence, once with the levers left None (values reach the
    shader as uniforms) and once with every hero carrying that same value
    explicitly (values reach it through the binding-5 row). Byte-identical is the
    only legitimate outcome — same numbers, different transport.

    Contention is load-bearing: these two heroes overlap, so heroRelaxWeight's
    max() actually compares candidates from different heroes and the `_e`
    bookkeeping is exercised. With a far-apart pair the max never contends."""
    g = _emergent_kin(taper=0.8).storms
    p_none = _emergent_kin(taper=0.8)
    p_none.storms.cast = [_hero(*_PAIR[0]), _hero(*_PAIR[1])]

    p_match = _emergent_kin(taper=0.8)
    p_match.storms.cast = [
        _hero(*pos, emergence=g.hero_emergence, shape=g.hero_shape,
              taper=g.hero_taper)
        for pos in _PAIR
    ]

    np.testing.assert_array_equal(_tracers(p_none, gpu), _tracers(p_match, gpu))


def test_emergence_is_per_storm(gpu):
    """Swapping WHICH of two heroes is emergent changes the output — impossible if
    emergence were still one global uniform (both scenes share the same global, the
    same two hero positions and the same set of override values)."""
    def scene(first: float, second: float) -> np.ndarray:
        p = _emergent_kin()
        p.storms.cast = [
            _hero(*_PAIR[0], emergence=first),
            _hero(*_PAIR[1], emergence=second),
        ]
        return _tracers(p, gpu).astype(np.float64)

    assert np.abs(scene(0.9, 0.0) - scene(0.0, 0.9)).max() > 1e-3


def test_opted_out_hero_does_not_touch_its_neighbours_relax_weight(gpu):
    """A hero at emergence 0 opted OUT: it must not reach heroRelaxWeight's
    accumulators at all. Its scaled candidate is 0, which TIES the initial
    `_e * acc` of 0 — without the loop-top guard the raw tie-break hands it the
    slot, and its RAW wake window then rides `flush *= 1.0 - wrel` into the
    EMERGENT neighbour's flush (and its zero `_e` zeroes the neighbour's `infl`).

    So: an emergent hero alone must render identically to the same hero with an
    opted-out neighbour ADDED far enough away to have no stamp of its own inside
    the emergent hero's neighborhood... which is not achievable while keeping the
    windows overlapping. Instead assert the invariant that IS separable: moving
    the opted-out neighbour's own levers (which only its stamp reads) must not
    disturb the emergent hero's relax weight, i.e. the two scenes differ only
    where the opted-out hero itself is stamped."""
    def scene(neighbour_taper: float) -> np.ndarray:
        p = _emergent_kin(taper=0.8)
        p.storms.cast = [
            _hero(*_PAIR[0], emergence=0.9),
            _hero(*_PAIR[1], emergence=0.0, taper=neighbour_taper),
        ]
        return _tracers(p, gpu).astype(np.float64)

    # taper is emergence-scaled, so on an emergence-0 hero it is inert
    # EVERYWHERE — including through the accumulators it must not reach.
    np.testing.assert_array_equal(scene(0.0), scene(1.4))


def test_cast_only_emergence_selects_the_variant(gpu):
    """Predicate gate: with the GLOBAL emergence at 0, a cast hero's own
    emergence override must still select HERO_EMERGENCE (dual-gated with
    CAST_LEVERS, which its own presence satisfies). Off-by-predicate would
    silently render the legacy stamped hero, so compare against emergence 0."""
    p_off = _emergent_kin(0.0)
    p_off.storms.cast = [_hero(-20.0, 0.0, emergence=0.0)]
    p_on = _emergent_kin(0.0)
    p_on.storms.cast = [_hero(-20.0, 0.0, emergence=0.9)]

    assert np.abs(_tracers(p_on, gpu).astype(np.float64)
                  - _tracers(p_off, gpu).astype(np.float64)).max() > 1e-3


@pytest.mark.parametrize("lever", ["shape", "taper"])
def test_shape_and_taper_are_per_storm(gpu, lever):
    """shape (vec4_2.y) and taper (vec4_2.z) ride their own columns and their own
    read sites (stamp anatomy, heroRelaxWeight's Rr/twr, heroBandDeflect's hold).
    Swap ONE at a time between two equally-emergent heroes: driving both together
    would still differ if the two columns were SWAPPED, or if one of them were
    never read at all — which is exactly the defect a column test must catch."""
    def scene(first: float, second: float) -> np.ndarray:
        p = _emergent_kin()
        p.storms.cast = [
            _hero(*_PAIR[0], emergence=0.9, **{lever: first}),
            _hero(*_PAIR[1], emergence=0.9, **{lever: second}),
        ]
        return _tracers(p, gpu).astype(np.float64)

    assert np.abs(scene(1.4, 0.0) - scene(0.0, 1.4)).max() > 1e-3


def _omega_scene(solid: float, **overrides) -> PlanetParams:
    """Vorticity dev-0 scene, both cast heroes at the same solid_core, each
    carrying whatever emergence-family overrides the caller passes as
    ``{lever}`` / ``{lever}2`` pairs."""
    p = _vort_params(solid_core=solid, second_solid_core=solid)
    p.storms.hero_emergence = 0.9
    for name, value in overrides.items():
        idx = 1 if name.endswith("2") else 0
        setattr(p.storms.cast[idx], name.rstrip("2"), value)
    return p


def test_per_storm_emergence_on_omega_is_a_noop_without_solid_core(gpu):
    """Documented coupling, both directions, on the byte-exact dev-0 omega:

    (a) with solid_core 0 on both heroes the omega side never enters the ring
        branch, so per-storm emergence cannot move omega -> byte-identical;
    (b) with solid_core on, swapping which hero is emergent DOES move it.

    Named `..._is_a_noop_...` deliberately: gpu-smoke selects on
    "identical or noop or no_op", and (a) is the branch's only byte-exact
    assertion on the omega path — off the blocking job it would run nightly
    only."""
    def omega(solid, e_first, e_second):
        return _dev0_omega(_omega_scene(solid, emergence=e_first,
                                        emergence2=e_second), gpu)

    np.testing.assert_array_equal(omega(0.0, 0.9, 0.0), omega(0.0, 0.0, 0.9))
    assert np.abs(omega(0.9, 0.9, 0.0) - omega(0.9, 0.0, 0.9)).max() > 1e-3


@pytest.mark.parametrize("lever", ["shape", "taper"])
def test_shape_and_taper_are_per_storm_on_omega(gpu, lever):
    """vortex_omega.glsl hoists shape (col 9) and taper (col 10) with their OWN
    read sites (the ring Rrs deformation + the tcomp circulation compensation).
    The kinematic swap tests cannot reach them — solid_core is a documented
    kinematic no-op — so without this a stale global or a swapped column on the
    omega side fails nothing."""
    def omega(first, second):
        return _dev0_omega(
            _omega_scene(0.9, **{lever: first, f"{lever}2": second}), gpu)

    assert np.abs(omega(1.4, 0.0) - omega(0.0, 1.4)).max() > 1e-3


def test_same_hero_spelled_two_ways_renders_the_same(gpu):
    """The regression test for the M2-B review's Critical. `_hero_emergence_active`
    learned to select the variant for a cast-only emergent hero, which made a
    state reachable that never was before: global emergence 0 with a hero
    emergent on its own override. omega_force's 60x core anchor and its wake
    eddy-injection still scaled by the raw GLOBAL uniform, so in that state both
    silently vanished — the SAME storm spelled two equivalent ways rendered 0.86
    apart in vorticity mode (86x GPU_NOISE_ATOL). The uniform now carries
    VortexRegistry.scene_emergence (the strongest EFFECTIVE emergence), so the
    two spellings agree.

    Tolerance, not byte-equality: this runs the vorticity solver past step 0, so
    SOR noise applies (CLAUDE.md)."""
    def spelled(global_e: float, override_e: float | None) -> np.ndarray:
        p = _vort_params(solid_core=0.9, second_solid_core=0.9)
        p.sim.dev_steps = 40
        p.storms.hero_emergence = global_e
        p.storms.wake_turbulence = 1.0
        for entry in p.storms.cast:
            entry.emergence = override_e
        sim = Simulation(p, gpu)
        sim.run_to_completion(chunk=64)
        return np.asarray(
            sim.gpu.read_texture(sim.solver.equirect.tracers.cur)).astype(np.float64)

    from_global = spelled(0.9, None)     # inherits -> effective 0.9
    from_override = spelled(0.0, 0.9)    # cast-only -> effective 0.9
    assert np.abs(from_global - from_override).max() < 1e-2


def test_emergence_reaches_omega_force_per_storm(gpu):
    """M2-C: the omega_force hero sites (the 60x core anchor and the wake eddy
    injection) follow per storm, via heroAnchorBoost/heroWakeInject.

    ISOLATION — solid_core is 0 on both heroes, so the omega side never enters
    the ring branch and emergence cannot reach omega_INIT at all. Part (a)
    asserts that byte-exactly on the dev-0 texture, which is what makes part (b)
    mean anything: with the initial state provably identical, a difference after
    stepping can only have come from the per-step path. Without (a) this would
    be the kind of test that passes because SOMETHING differs.

    Scope, stated honestly: (b) pins that per-storm emergence reaches the
    stepping solver, which is omega_force's two sites plus psi's wake wedge --
    it does not separate those. The no-op direction for omega_force (the
    CAST_LEVERS-off arm reproducing the legacy lines bit-for-bit) is not
    assertable in-process at all and is gated by scripts/m2c_omega_equiv.py.

    Tolerance, not byte-equality, in (b): past step 0 the SOR noise applies."""
    def scene(e_a: float, e_b: float, steps: int) -> PlanetParams:
        p = _vort_params(solid_core=0.0, second_solid_core=0.0)
        p.sim.dev_steps = steps
        p.storms.hero_emergence = 0.9
        p.storms.wake_turbulence = 1.0
        p.storms.cast[0].emergence = e_a
        p.storms.cast[1].emergence = e_b
        return p

    # (a) the setup really is emergence-blind before any force pass runs
    np.testing.assert_array_equal(
        _dev0_omega(scene(0.9, 0.1, 0), gpu), _dev0_omega(scene(0.1, 0.9, 0), gpu)
    )

    # (b) ...and swapping which hero is emergent moves the stepped state
    def stepped(p: PlanetParams) -> np.ndarray:
        sim = Simulation(p, gpu)
        sim.run_to_completion(chunk=64)
        q = np.asarray(sim.gpu.read_texture(sim.solver._omega_state.cur))
        return np.squeeze(q).astype(np.float64)

    a = stepped(scene(0.9, 0.1, 8))
    b = stepped(scene(0.1, 0.9, 8))
    assert np.abs(a - b).max() > 1e-2

"""GPU tests for storms.hero_emergence (flow-negotiated hero edge).

The relaxation forcing (advect.comp pass 2) re-imposes the analytic hero stamp
every step, so the flow never owns the storm -> it reads as stamped. hero_emergence
fades the relaxation rate through the hero rim/collar/near-interior (heroRelaxWeight
in vortex_stamp.glsl), so advection folds ambient tracer there instead. The
deep-core anchor keeps full relaxation, and the weight is exactly 1.0 far from any
hero, so everything outside the storm neighborhood is byte-identical.

The whole feature compiles as a HERO_EMERGENCE preprocessor variant selected by
solver._domain_defines (emergence > 0 AND a hero exists), so "off is the
pre-feature program" is structural — pinned by the kinematic source hashes
(tests/unit/test_kinematic_kernels_pinned.py) and the p05 render-hash gate, not
re-provable at runtime. What CAN be pinned at runtime, and is below:
  1. the default kinematic path (emergence=0, rim levers on) is deterministic
     across full Simulation rebuilds;
  2. emergence>0 with NO hero selects the default program (predicate pin);
  3. with the variant COMPILED (hero present, emergence>0), the far field is
     byte-identical — the runtime forced-variant no-op, hero-locality edition;
  4. the anchored plateau fill actually lands on the registry position.

Every byte-exact assert here relies on the KINEMATIC solver path (the vorticity
SOR solve carries a documented ~1e-3 noise floor and is never byte-compared);
_params asserts the mode so a future default-solver flip fails loudly instead
of flaking against the noise floor.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _params(
    emergence: float = 0.0,
    hero_count: int = 1,
    rim_tint: float = 0.0,
    rim_warp: float = 0.0,
) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    p.storms.hero_count = hero_count
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_rim_tint = rim_tint
    p.storms.hero_rim_warp = rim_warp
    p.storms.hero_emergence = emergence
    # The byte-exact asserts in this file are only valid on the kinematic path
    # (vorticity output is tolerance-compared everywhere else in the suite).
    assert p.solver.type == SolverType.KINEMATIC
    return p


def _developed_tracers(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


# ------------------------------------------------------------- byte-identity

def test_emergence_off_byte_identical_with_other_levers_on(gpu):
    """Determinism canary for the emergence=0 program: both runs use IDENTICAL
    params (emergence defaults to 0.0 in _params — this is deliberate, not an
    off-vs-on comparison), so this pins that the default kinematic path with
    rim_tint + rim_warp on is reproducible across two full Simulation builds.
    The actual off == pre-feature guarantee is structural (variant not
    compiled) and is pinned by the source hashes + p05, not runtime-testable."""
    base = _developed_tracers(_params(rim_tint=0.7, rim_warp=0.5), gpu)
    same = _developed_tracers(_params(emergence=0.0, rim_tint=0.7, rim_warp=0.5), gpu)
    np.testing.assert_array_equal(base, same)


def test_emergence_no_hero_is_byte_identical(gpu):
    """With NO hero, _domain_defines does not select the HERO_EMERGENCE variant
    (the predicate requires a hero), so emergence>0 runs the DEFAULT program —
    byte-identical to off by construction. This pins the predicate: a no-hero
    config must never pay the variant's per-pixel vortex-SSBO scan for a
    guaranteed no-op (heroRelaxWeight would return exactly 1.0 everywhere)."""
    off = _developed_tracers(_params(emergence=0.0, hero_count=0), gpu)
    on = _developed_tracers(_params(emergence=1.0, hero_count=0), gpu)
    np.testing.assert_array_equal(off, on)


# ------------------------------------------------------------- effect + locality

def test_emergence_anchors_red_fill_on_hero(gpu):
    """Vorticity mode, emergence on: the hero anchor keeps the prognostic core
    glued to the registry position and the plateau fill paints it red, so the
    developed T3 tint AT the registry position must be strongly warm. (Without
    the anchor the core wanders ~0.2 rad from the stamp and the probe reads
    ~0.0 — the diagnostic that motivated the anchor.)"""
    from gasgiant.params.presets import load_factory_preset

    p = load_factory_preset("gas_giant_warm").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 300
    assert p.solver.type == SolverType.VORTICITY
    p.storms.hero_emergence = 1.0
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    tr = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)
    hero = sim.vortices.heroes()[0]
    h, w = tr.shape[:2]
    row = int((0.5 - hero.lat / np.pi) * h)
    col = int((hero.lon + np.pi) / (2.0 * np.pi) * w)
    # Average T3 over the interior (a few pixels around the center) to be robust
    # to per-pixel mottle; the plateau target is ~hero.tint (0.9).
    patch = tr[row - 2 : row + 3, [c % w for c in range(col - 3, col + 4)], 3]
    assert patch.mean() > 0.3, (
        f"hero interior T3 at the registry position is {patch.mean():.2f} — "
        "the anchored plateau fill did not land on the storm"
    )


def _solo_hero_params(**kw) -> PlanetParams:
    """_params variant with EVERY non-hero storm population zeroed, so the
    tracer field contains exactly one hero stamp on top of the bands — the
    geometry probes below difference/compare against the pure band field and
    any other seeded storm would pollute the windows. Longitude pinned to 0 so
    the sector math needs no wrap handling."""
    p = _params(**kw)
    p.storms.hero_longitude = 0.0
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.accent_count = 0
    p.storms.hero_companions = 0
    return p


def _sim_and_tracers(p: PlanetParams, gpu):
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    return sim, sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


def _lonlat_grids(shape):
    h, w = shape[:2]
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon = ((np.arange(w) + 0.5) / w) * 2.0 * np.pi - np.pi
    return np.meshgrid(lon, lat)


# ------------------------------------------- Phase-1 GRS-interaction behavior
# (plan ancient-snuggling-meadow: moat shear-asymmetry, belt bowing, wake
# release + forcing, extended locality)

def test_emergence_moat_asymmetry_carves_downstream_arc(gpu):
    """The moat shear-asymmetry is deterministic and wake-keyed: the collar's
    DOWNSTREAM arc (the side wake_dir points to — flow-derived under
    emergence, see _hero_wake_frame) is carved down, the upstream arc is not.
    Probed at dev_steps=0 (the pure stamp) on the SOUTH half of the collar
    annulus, where the two sectors sample identical latitudes (band background
    cancels in the comparison) and the wake wedge's own stamp is negligible
    (|across| >= 1 and it decays as exp(-across^2))."""
    p = _solo_hero_params(emergence=0.9)
    p.sim.dev_steps = 0
    p.storms.rim_contrast = 1.5
    p.bands.warp_amount = 0.0
    sim, tr = _sim_and_tracers(p, gpu)
    hero = sim.vortices.heroes()[0]

    lon_g, lat_g = _lonlat_grids(tr.shape)
    dlat = lat_g - hero.lat
    dlon = lon_g - hero.lon
    x_east = dlon * np.cos(hero.lat) / hero.aspect
    q = np.hypot(x_east, dlat) / hero.r_core
    hth = np.arctan2(dlat, x_east)   # 0 = east, +pi/2 = north, +-pi = west
    # Probe in the DEFORMED outline frame: the shape deformation (equatorward
    # flatten + seeded m=2/3 breathing, storms.hero_shape) moves the collar's
    # test-frame radius by up to ~9% per azimuth, so a rigid-ellipse annulus
    # catches different anatomy per azimuth and dilutes the carve margin
    # (measured 0.02 -> 0.008). Mirror of the shader's R(theta)
    # (vortex_stamp.glsl; this hth equals the shader's PI - hth, and the
    # phases are the solver's dedicated hero-shape substream); e = 0.9 and
    # hero_shape = default 1.0 from the config above. The pinned kernel hash
    # forces a conscious update here if the shader constants move.
    sph = np.asarray(sim.solver._shape_phase)
    neq = np.maximum(np.sin(hth), 0.0)   # southern hero: equatorward = north
    rr = 1.0 - p.storms.hero_shape * 0.9 * (0.11 * neq * neq
                                            - 0.05 * np.sin(2.0 * hth + sph[0])
                                            - 0.04 * np.sin(3.0 * hth + sph[1]))
    q = q / rr

    annulus = (q > 1.15) & (q < 1.6)
    south = np.sin(hth) < -0.3
    # Test hth: 0 = east. Downstream = the direction wake_dir points.
    down = annulus & south & (np.cos(hth) * hero.wake_dir > 0.5)
    up = annulus & south & (np.cos(hth) * hero.wake_dir < -0.5)
    t0 = tr[..., 0]
    assert t0[down].mean() < t0[up].mean() - 0.02, (
        f"downstream collar arc (mean {t0[down].mean():.3f}) is not carved "
        f"below the upstream arc (mean {t0[up].mean():.3f})"
    )


def test_emergence_bows_band_boundary_around_hero(gpu):
    """heroBandDeflect: a band boundary crossing the hero's collar zone must
    bow AWAY from the hero by >= 0.8 r_core at the hero longitude (the
    reference's tight-but-strong Hollow; the 0.3-r_core bow of the first spec
    draft was rejected in plan review) and recover to the straight boundary
    far away. Stamp rings are disabled (rim_contrast=0) and the band warp is
    zeroed so the iso-crossing is pure band signal. Also pins determinism of
    the emergence=0 program at the belt-edge placement (two identical runs)."""
    # Stage 1: find the strongest band boundary in the hero's latitude range,
    # and the seeded (jittered) core radius, with the deflection OFF.
    p = _solo_hero_params(emergence=0.0)
    p.sim.dev_steps = 0
    p.storms.rim_contrast = 0.0
    p.bands.warp_amount = 0.0
    sim, tr = _sim_and_tracers(p, gpu)
    r = sim.vortices.heroes()[0].r_core
    h, w = tr.shape[:2]
    lat_axis = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    far = tr[:, w // 8 : 3 * w // 8, 0].mean(axis=1)     # quarter turn away
    search = (lat_axis > np.radians(-35.0)) & (lat_axis < np.radians(-8.0))
    grad = np.abs(np.gradient(far))
    grad[~search] = 0.0
    b_row = int(np.argmax(grad))
    lat_b = lat_axis[b_row]

    # Stage 2: park the hero so the boundary sits at ~0.45 r_core from its
    # center (inside the deflection window; the self-consistent bow there is
    # ~0.93 r_core at e=0.9). Latitude pinning is applied AFTER the RNG draw,
    # so r_core is unchanged. Emergence 0: boundary must stay straight; two
    # runs must be byte-identical (determinism at the new placement).
    def _placed(e: float) -> PlanetParams:
        q = _solo_hero_params(emergence=e)
        q.sim.dev_steps = 0
        q.storms.rim_contrast = 0.0
        q.bands.warp_amount = 0.0
        q.storms.hero_latitude = float(np.degrees(lat_b - 0.45 * r))
        return q

    sim0, off = _sim_and_tracers(_placed(0.0), gpu)
    assert abs(sim0.vortices.heroes()[0].r_core - r) < 1e-9
    _, off2 = _sim_and_tracers(_placed(0.0), gpu)
    np.testing.assert_array_equal(off, off2)

    _, on = _sim_and_tracers(_placed(0.9), gpu)

    # Boundary side values from the far-longitude profile, sampled clear of
    # the transition; the crossing level is their midpoint.
    north_val = far[max(b_row - int(0.6 * r * h / np.pi), 1)]
    south_val = far[min(b_row + int(0.6 * r * h / np.pi), h - 1)]
    mid = 0.5 * (north_val + south_val)

    def crossings(tracers: np.ndarray, col: int) -> np.ndarray:
        """Latitudes where T0 crosses mid inside (lat_b - 0.5 r, lat_b + 1.6 r)
        — wide enough for both the straight boundary (at lat_b) and the bowed
        one (expected ~lat_b + 0.93 r), tight enough to exclude the next
        template edge (the default template has one ~2 r north)."""
        prof = tracers[:, col, 0]
        win = (lat_axis > lat_b - 0.5 * r) & (lat_axis < lat_b + 1.6 * r)
        rows = np.where(win)[0]
        sign = prof[rows] > mid
        flips = np.where(sign[:-1] != sign[1:])[0]
        assert flips.size, "no boundary crossing found in the probe window"
        return lat_axis[rows[flips]]

    hero_col = w // 2  # hero_longitude pinned to 0
    far_col = (hero_col + w // 4) % w
    # Stage-1 isolation: the chosen boundary must be the ONLY crossing in the
    # probe window far from the hero, or nearest/northmost selection below is
    # ambiguous (another template edge inside +2 r would pollute it).
    assert crossings(off, far_col).size == 1, (
        "band template has another boundary within 2 r of the chosen one — "
        "probe window ambiguous"
    )
    l_off = float(crossings(off, hero_col)[np.argmin(np.abs(crossings(off, hero_col) - lat_b))])
    l_far = float(crossings(on, far_col)[np.argmin(np.abs(crossings(on, far_col) - lat_b))])
    l_on = float(crossings(on, hero_col).max())   # northmost = the bowed apex

    row_lat = np.pi / h
    assert abs(l_off - lat_b) < 2.5 * row_lat, "boundary moved with emergence OFF"
    assert abs(l_far - lat_b) < 2.5 * row_lat, "boundary deflected far from the hero"
    assert (l_on - lat_b) >= 0.8 * r, (
        f"belt boundary bowed only {np.degrees(l_on - lat_b):.2f} deg "
        f"({(l_on - lat_b) / r:.2f} r_core) at the hero longitude — "
        "the tight-but-strong Hollow needs >= 0.8 r_core"
    )


def test_emergence_wake_sector_folds_downstream_only(gpu):
    """Vorticity mode: the wake forcing (omega_force wedge injection) plus the
    relaxation release (heroRelaxWeight) must leave persistent folded tracer
    structure DOWNSTREAM (west) of the hero that the upstream side does not
    have — the reference wake asymmetry. Belt-straddling placement per plan
    review: at the zone-centered legacy latitude the wedge sits in uniform
    zone material and this probe would only measure noise. Tolerance-based
    (vorticity path, never byte-compared); other storm populations are zeroed
    so the two windows compare hero physics, not seeded oval placement."""
    from gasgiant.params.presets import load_factory_preset

    p = load_factory_preset("gas_giant_warm").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 300
    assert p.solver.type == SolverType.VORTICITY
    p.storms.hero_latitude = -21.0
    p.storms.hero_longitude = 0.0
    p.storms.hero_emergence = 0.9
    p.storms.wake_turbulence = 3.2
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.accent_count = 0
    p.storms.hero_companions = 0
    # The background SCENE is part of this test's premise and is FROZEN:
    # small storms + outbreaks stay at the values the E/W statistic was
    # calibrated against (they shape the chaotic vorticity field everywhere,
    # so ANY registry change re-rolls the fold pattern — zeroing them parked
    # a belt-edge inject eddy in the upstream window; the round-B density
    # bump to 3.5 dropped a small storm there). Pinning them makes the probe
    # immune to preset population drift without re-rolling the scene.
    p.storms.small_density = 3.0
    p.storms.outbreak_count = 2
    # hero_shape deforms the vorticity ring/skirt (the streamlines) — part of
    # the frozen scene contract: with it live, every shape retune re-rolls
    # the chaotic fold pattern and the E/W margin with it. Wake physics is
    # orthogonal to the outline deformation; pin the exact analytic oval.
    p.storms.hero_shape = 0.0
    sim, tr = _sim_and_tracers(p, gpu)
    hero = sim.vortices.heroes()[0]

    lon_g, lat_g = _lonlat_grids(tr.shape)
    dlon = np.mod(lon_g - hero.lon + 3.0 * np.pi, 2.0 * np.pi) - np.pi
    an = dlon * hero.wake_dir / hero.r_core          # + = downstream (west)
    across = (lat_g - (hero.lat + hero.wake_lat_off)) / (hero.r_core * 1.8)

    def hp_std(mask: np.ndarray) -> float:
        """Row-mean-removed std of T0 inside the mask (kills the band
        gradient so the statistic measures FOLDS, not banding)."""
        t0 = tr[..., 0]
        vals = []
        for row in np.unique(np.where(mask)[0]):
            v = t0[row][mask[row]]
            if v.size > 4:
                vals.append(v - v.mean())
        return float(np.concatenate(vals).std())

    # Windows start at an=4 (q ~ 1.8 at aspect 2.2): the diffuse dark collar
    # (ring_q 1.30, k 12) has a stamped radial tail to q ~ 1.7 that the
    # row-mean removal cannot kill (q varies along a row), and a window
    # overlapping it measures anatomy, not folds.
    lane = np.abs(across) < 1.2
    wake_std = hp_std(lane & (an > 4.0) & (an < 7.0))
    up_std = hp_std(lane & (an < -4.0) & (an > -7.0))
    assert wake_std > 1.3 * up_std, (
        f"wake sector fold variance ({wake_std:.4f}) does not exceed the "
        f"upstream sector ({up_std:.4f}) — the wake is not folding downstream"
    )


# ------------------------------------------- Round-B behavior (same plan:
# de-bullseye flush pinch, interior T3 banding)

def test_emergence_flush_pinches_belt_side(gpu):
    """The meridionally shaped flush must erase wound deviations HARDER on the
    equatorward (belt) side than poleward — the reference's hollow is pinched
    hard against the belt, and a radially-uniform flush halo was the bullseye's
    outermost ring. Probed on the developed kinematic field as the residual
    from the far-longitude zonal profile in the q 1.6-2.0 shell (where the
    poleward inner rise 1.66 vs the belt-side 1.19 gives the maximum flush
    differential). Probe hygiene, both learned from a first failing draft:
    the band VALUE contrast is dropped to 0.08 — below the CPU bow gate's
    0.04-step floor over +-1.6 r, so heroBandDeflect is OFF by construction
    (asserted on the registry; at a gated placement the designed bow
    dominates the equatorward residual and the statistic reads a feature as
    un-flushed material) while ~8% of the latitudinal T0 gradient survives
    as wound-material signal (the residual scales linearly, the ratio does
    not); and the shell is restricted to the strictly-UPSTREAM half (an < 0)
    because the stamped wake wedge is active from along = 0."""
    p = _solo_hero_params(emergence=0.9)
    p.sim.dev_steps = 150
    p.bands.warp_amount = 0.0
    p.bands.value_contrast = 0.08
    # Radius 0.05: at the default 0.10 the gate's +-1.6 r window spans ~19 deg
    # = several bands, and even 8% contrast accumulates a step past the gate
    # floor. One band's worth of window keeps the flattened step under it.
    p.storms.hero_radius = 0.05
    sim, tr = _sim_and_tracers(p, gpu)
    hero = sim.vortices.heroes()[0]
    h, w = tr.shape[:2]
    assert hero.bow_gain == 0.0, (
        "bow gate engaged despite the flattened bands — the probe would read "
        "the designed belt bow as residual (did the gate thresholds change?)"
    )

    lon_g, lat_g = _lonlat_grids(tr.shape)
    dlat = lat_g - hero.lat
    dlon = np.mod(lon_g - hero.lon + 3.0 * np.pi, 2.0 * np.pi) - np.pi
    x_east = dlon * np.cos(hero.lat) / hero.aspect
    q = np.hypot(x_east, dlat) / hero.r_core
    m = (dlat / hero.r_core) / np.maximum(q, 1e-9)   # sin(hero azimuth), N=+
    an = dlon * hero.wake_dir / hero.r_core          # + = downstream

    ref = tr[:, w // 8 : 3 * w // 8, 0].mean(axis=1)  # quarter turn away
    resid = np.abs(tr[..., 0] - ref[:, None])

    shell = (q > 1.6) & (q < 2.0) & (an < 0.0)
    # Hero is in the southern hemisphere: equatorward = north = m > 0 (matches
    # the shader's eqs sign from the center-y hemisphere test).
    eq_res = resid[shell & (m > 0.8)].mean()
    pol_res = resid[shell & (m < -0.8)].mean()
    assert pol_res > 3e-5, (
        "poleward shell residual is ~zero — the probe premise broke (nothing "
        "left to compare; did the flush windows or dev_steps change?)"
    )
    assert eq_res < 0.75 * pol_res, (
        f"equatorward shell residual ({eq_res:.4f}) is not pinched below the "
        f"poleward residual ({pol_res:.4f}) — the belt-side flush pinch is "
        "not asserting the band harder than the zone-side moat"
    )


def test_hero_shape_lever_and_seed(gpu):
    """storms.hero_shape / hero_shape_seed (the outline-deformation lever):
    shape=0 is a deterministic exact oval distinct from shape=1; the seed
    re-rolls the lobes on its own substream (different outline, same
    everything else at stamp level); same seed reproduces bit-for-bit.
    Kinematic path (byte-exact asserts legal)."""
    def render(shape: float, seed: int) -> np.ndarray:
        p = _solo_hero_params(emergence=0.9)
        p.sim.dev_steps = 0
        p.storms.hero_shape = shape
        p.storms.hero_shape_seed = seed
        return _sim_and_tracers(p, gpu)[1]

    oval = render(0.0, 0)
    oval2 = render(0.0, 0)
    np.testing.assert_array_equal(oval, oval2)          # determinism at 0

    egg = render(1.0, 0)
    assert np.abs(egg - oval).max() > 1e-3, "hero_shape=1 did not deform"

    egg_b = render(1.0, 7)
    assert np.abs(egg_b - egg).max() > 1e-3, "hero_shape_seed did not re-roll"
    # The seed must be inert while the deformation is OFF (own substream,
    # consumed only by the shape terms).
    oval_b = render(0.0, 7)
    np.testing.assert_array_equal(oval_b, oval)


def test_emergence_interior_t3_banding(gpu):
    """Interior circulation legibility rides T3 (the |T3|~0.9 tint blend swamps
    T0 — measured root cause): the T3 spiral lane + knot + nucleus must give
    the pure stamp visible interior T3 structure, while the anchored plateau
    keeps its strongly-warm MEAN (co-pin of the anchor property at stamp level
    so the banding can never be 'satisfied' by washing the interior out).
    Mottle and tint_var are zeroed so only the deterministic structure terms
    contribute."""
    p = _solo_hero_params(emergence=0.9)
    p.sim.dev_steps = 0
    p.storms.hero_mottle = 0.0
    p.storms.hero_tint_var = 0.0
    sim, tr = _sim_and_tracers(p, gpu)
    hero = sim.vortices.heroes()[0]

    lon_g, lat_g = _lonlat_grids(tr.shape)
    dlat = lat_g - hero.lat
    dlon = np.mod(lon_g - hero.lon + 3.0 * np.pi, 2.0 * np.pi) - np.pi
    x_east = dlon * np.cos(hero.lat) / hero.aspect
    q = np.hypot(x_east, dlat) / hero.r_core

    t3 = tr[..., 3][(q > 0.35) & (q < 0.8)]
    assert t3.std() > 0.04, (
        f"interior T3 std {t3.std():.4f} — the spiral banding / knot / nucleus "
        "are not producing visible interior tint structure"
    )
    assert t3.mean() > 0.3, (
        f"interior T3 mean {t3.mean():.2f} — the banding dips washed out the "
        "plateau's warm identity"
    )


def test_emergence_locality_far_south(gpu):
    """Extended locality (plan review: the far-north check alone would miss a
    SOUTHERN leak from the new wake windows, which reach equatorward AND
    poleward of the hero). Every new window's reach tops out at ~-47 deg at
    max radius jitter, and the semi-Lagrangian domain of dependence spreads
    ~0.5 cell/step (~10 deg over the 60-step run), so south of -67.5 deg is
    provably untouched (21 deg margin; the polar confinement band past 60 deg
    damps flow differences on top). NOTE an upstream-COLUMNS byte-identity check
    is deliberately absent: emergence changes the flow (the extended psi
    wedge), and tracer differences advect zonally around the whole hero
    latitude band — east-side cleanliness is a stamp/relax-target property,
    not a developed-field one. The wake's east/west ASYMMETRY is asserted
    tolerance-based in test_emergence_wake_sector_folds_downstream_only."""
    base = _solo_hero_params(emergence=0.0)
    on_p = _solo_hero_params(emergence=0.8)
    off = _developed_tracers(base, gpu)
    on = _developed_tracers(on_p, gpu)
    h = off.shape[0]

    far_south = slice(int(0.875 * h), h)   # south of -67.5 deg
    np.testing.assert_array_equal(on[far_south], off[far_south])


def test_emergence_changes_hero_neighborhood_only(gpu):
    """The runtime forced-variant no-op test (CLAUDE.md lever rule), hero-local
    edition: with a hero present and emergence>0 the HERO_EMERGENCE variant IS
    compiled, and it must (a) measurably change the developed tracers near the
    hero (the relaxation there is faded so advection folds the field) while
    (b) leaving the far field byte-identical — heroRelaxWeight returns exactly
    1.0 outside q<3.6, so rk == u_relax_k bit-for-bit out there."""
    off = _developed_tracers(_params(emergence=0.0), gpu)
    on = _developed_tracers(_params(emergence=0.8), gpu)

    # SOME change near the hero (T0 brightness), past the vorticity noise floor.
    delta = np.abs(on[..., 0] - off[..., 0])
    assert delta.max() > 1e-2, "hero_emergence did not change the hero neighborhood"

    # Locality: the far NORTH quarter (hero is at -22.5 deg south, i.e. the
    # southern half) is byte-identical. heroRelaxWeight returns exactly 1.0 there
    # (no hero within q<3.6), so rk is unchanged and the relaxation math matches
    # bit-for-bit. A real leak would be obvious given the hero is far south.
    h = off.shape[0]
    far = slice(0, h // 4)                       # top quarter = far north
    np.testing.assert_array_equal(on[far], off[far])

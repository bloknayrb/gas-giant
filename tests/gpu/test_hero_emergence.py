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
                                            - 0.075 * np.sin(2.0 * hth + sph[0])
                                            - 0.055 * np.sin(3.0 * hth + sph[1]))
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
    structure DOWNSTREAM of the hero (east on this scene — the axis follows
    hero.wake_dir, flow-derived under emergence) that the upstream side does
    not have — the reference wake asymmetry. Belt-straddling placement per plan
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
    # jets.local_jet_speed pinned OFF: the chirality fix (co-rotate all
    # storms with ambient shear) bakes a local westward jet into warm
    # (-0.9 @ -20.0 w0.05) so the -22.0 hero lands in strong co-rotating
    # shear, but THIS probe's hero_latitude stays frozen at the pre-bake
    # -21.0 (see below) -- 1 deg from the jet center, inside its search
    # window for _hero_wake_frame. With the jet live the probe measured
    # 1.28 (fails the 1.3 gate); with it neutralized, matching the ambient
    # profile the gate was calibrated against, seed 7 measures 1.355 (was
    # 1.334 pre-flip) and 3/4 probe seeds {7,11,23,42} pass both before and
    # after the flip (11 flips fail->pass, 23 flips pass->fail — the
    # statistic was already seed-sensitive pre-flip, not newly so). The
    # chirality flip itself does not weaken wake folding; the unpinned new
    # lever did (fix/vortex-chirality, commit 4b60fa6).
    p.jets.local_jet_speed = 0.0
    # Hero jet BRACKET pinned OFF (2026-07-19 GRS bake): warm now bakes an active
    # carve-and-impose bracket (hero_bracket_north/south) that supersedes the old
    # local_jet. It reorganizes the jets around the hero, so a live bracket
    # re-rolls this frozen fold pattern exactly as local_jet did -- pin it off to
    # keep the pre-bake ambient the E/W statistic was calibrated against.
    p.jets.hero_bracket_north = 0.0
    p.jets.hero_bracket_south = 0.0
    # The background SCENE is part of this test's premise and is FROZEN:
    # small storms + outbreaks stay at the values the E/W statistic was
    # calibrated against (they shape the chaotic vorticity field everywhere,
    # so ANY registry change re-rolls the fold pattern — zeroing them parked
    # a belt-edge inject eddy in the upstream window; the round-B density
    # bump to 3.5 dropped a small storm there). Pinning them makes the probe
    # immune to preset population drift without re-rolling the scene.
    p.storms.small_density = 3.0
    p.storms.outbreak_count = 2
    # hero_shape/hero_taper deform the vorticity ring/skirt (the streamlines)
    # — part of the frozen scene contract: with them live, every shape retune
    # re-rolls the chaotic fold pattern and the E/W margin with it. The taper
    # pin matters doubly: its wedge lives ON the upstream arc this statistic
    # samples. Wake physics is orthogonal to the outline deformation; pin the
    # exact analytic oval.
    p.storms.hero_shape = 0.0
    p.storms.hero_taper = 0.0
    # hero_flow_aspect widens the ring/skirt (the streamlines) too. This pin
    # is LOAD-BEARING at the preset bake: the scene loads warm, and a baked
    # K > 1 would re-roll the whole frozen fold pattern without it.
    p.storms.hero_flow_aspect = 1.0
    # hero_aspect shapes the ring/skirt AND this probe's own geometry (the
    # window-start comment above derives q from aspect 2.2), so it is part of
    # the frozen scene contract too: pinned at the pre-aspect-pass value so
    # the warm bake of 2.9 (2026-07-16) does not re-roll the fold pattern.
    p.storms.hero_aspect = 2.2
    # Every remaining stamp-side lever that shapes the relax target is pinned
    # at its current warm bake (PR-43 review: the contract's immunity to
    # preset drift was only partial — the NEXT warm retune of any of these
    # would re-roll the chaotic fold pattern into a confusing far-field red).
    p.storms.hero_radius = 0.062
    p.storms.rim_contrast = 1.3
    p.storms.hero_mottle = 0.9
    p.storms.hero_rim_warp = 1.0
    p.storms.hero_wake_detail = 1.0
    sim, tr = _sim_and_tracers(p, gpu)
    hero = sim.vortices.heroes()[0]

    lon_g, lat_g = _lonlat_grids(tr.shape)
    dlon = np.mod(lon_g - hero.lon + 3.0 * np.pi, 2.0 * np.pi) - np.pi
    an = dlon * hero.wake_dir / hero.r_core          # + = downstream (wake_dir)
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


def test_hero_taper_lever(gpu):
    """storms.hero_taper (the upstream wedge): taper=0 is deterministic
    (byte-equal reruns — determinism, NOT a no-op claim: the restructured
    guard's no-op is proven by the cross-commit capture, see the S1 commit);
    taper=1 deforms; it works with the lobes at 0 AND composes with them.
    Kinematic dev-0 path (byte-exact asserts legal)."""
    def render(shape: float, taper: float) -> np.ndarray:
        p = _solo_hero_params(emergence=0.9)
        p.sim.dev_steps = 0
        p.storms.hero_shape = shape
        p.storms.hero_taper = taper
        return _sim_and_tracers(p, gpu)[1]

    base = render(0.0, 0.0)
    base2 = render(0.0, 0.0)
    np.testing.assert_array_equal(base, base2)          # determinism at 0

    wedge = render(0.0, 1.0)
    assert np.abs(wedge - base).max() > 1e-3, "hero_taper=1 did not deform"

    lobes = render(1.0, 0.0)
    both = render(1.0, 1.0)
    assert np.abs(both - lobes).max() > 1e-3, "taper inert on top of the lobes"
    assert np.abs(both - wedge).max() > 1e-3, "lobes inert on top of the taper"


def test_hero_taper_is_upstream_wedge(gpu):
    """The taper is a wdir-keyed UPSTREAM wedge: at dev 0 (pure stamp; lobes
    and rim warp zeroed so the outline is otherwise analytic) the tapered
    stamp differs from the untapered one ONLY on the upstream half — the
    wedge weight 6.75*c^4*(1-c^2) with c = max(upstream_cos, 0) is exactly 0
    downstream, and there Rr stays exactly 1.0 (q /= 1.0 is IEEE-exact), so
    downstream texels are BYTE-identical. The bite lands in the ~35 deg
    shoulder band, and both forced wake directions place it on their own
    upstream side (the mirror pin: wdir is the one frame ingredient the
    seeded lobes never used)."""
    from gasgiant.params.model import WakeDir

    def render(taper: float, wake: WakeDir):
        p = _solo_hero_params(emergence=0.9, rim_warp=0.0)
        p.sim.dev_steps = 0
        p.storms.hero_shape = 0.0
        p.storms.hero_taper = taper
        p.storms.hero_wake_dir = wake
        return _sim_and_tracers(p, gpu)

    for wake in (WakeDir.WEST, WakeDir.EAST):
        sim0, base = render(0.0, wake)
        _, tap = render(1.0, wake)
        hero = sim0.vortices.heroes()[0]
        assert hero.wake_dir == (-1.0 if wake == WakeDir.WEST else 1.0)

        lon_g, lat_g = _lonlat_grids(base.shape)
        dlat = lat_g - hero.lat
        dlon = np.mod(lon_g - hero.lon + 3.0 * np.pi, 2.0 * np.pi) - np.pi
        x_east = dlon * np.cos(hero.lat) / hero.aspect
        q = np.hypot(x_east, dlat) / hero.r_core
        # Upstream-signed squashed cosine — the same construction as the
        # shader's uct (test frame vs shader chord metric differ at O(q^2)
        # near the storm; the mask margins below absorb that).
        uct = -hero.wake_dir * x_east / np.maximum(hero.r_core * q, 1e-9)
        diff = np.abs(tap.astype(np.float64) - base.astype(np.float64)).max(axis=-1)

        down = uct < -0.02
        assert diff[down].max() == 0.0, (
            f"wake={wake}: taper leaked onto the downstream half "
            f"(max diff {diff[down].max():.3e})"
        )
        shoulder = (uct > 0.6) & (uct < 0.95) & (q > 0.3) & (q < 2.0)
        assert diff[shoulder].max() > 1e-3, (
            f"wake={wake}: no wedge bite in the upstream shoulder band"
        )


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
    1.0 outside its q 4.2 cull, so rk == u_relax_k bit-for-bit out there."""
    off = _developed_tracers(_params(emergence=0.0), gpu)
    on = _developed_tracers(_params(emergence=0.8), gpu)

    # SOME change near the hero (T0 brightness), past the vorticity noise floor.
    delta = np.abs(on[..., 0] - off[..., 0])
    assert delta.max() > 1e-2, "hero_emergence did not change the hero neighborhood"

    # Locality: the far NORTH quarter (hero is at -22.5 deg south, i.e. the
    # southern half) is byte-identical. heroRelaxWeight returns exactly 1.0 there
    # (no hero within its q 4.2 cull), so rk is unchanged and the relaxation math
    # matches bit-for-bit. A real leak would be obvious given the hero is far south.
    h = off.shape[0]
    far = slice(0, h // 4)                       # top quarter = far north
    np.testing.assert_array_equal(on[far], off[far])


# ------------------------------------------------------------ hero_flow_aspect

def _solo_warm_params(**storms_over) -> PlanetParams:
    """gas_giant_warm — VORTICITY: hero_flow_aspect lives on the omega path
    only (vortex_omega.glsl is compiled by omega_init/omega_force alone, so
    the lever cannot touch kinematic output at all) — with every non-hero
    population zeroed and the hero pinned, for omega-texture geometry probes
    against the pure jet background."""
    from gasgiant.params.presets import load_factory_preset

    p = load_factory_preset("gas_giant_warm").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_latitude = -21.0
    p.storms.hero_longitude = 0.0
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.accent_count = 0
    p.storms.hero_companions = 0
    p.storms.small_density = 0.0
    p.storms.outbreak_count = 0
    p.storms.hero_shape = 0.0
    p.storms.hero_taper = 0.0
    # Frozen probe GEOMETRY (2026-07-19 GRS bake): these omega geometry probes are
    # calibrated at hero_latitude -21 / hero_aspect 2.2 / r_core 0.062 (the
    # pre-bake values the ratio/centroid masters were measured at -- see the
    # widens-test comments). The bake moved warm's hero to -24 / aspect 2.0 /
    # r 0.108, so pin the calibration geometry here (the measurement boxes scale
    # with aspect*r_core; letting them track the bake re-rolls the masters).
    p.storms.hero_aspect = 2.2
    p.storms.hero_radius = 0.062
    # Bracket pinned OFF (same bake): warm now bakes an active bracket that
    # reorganizes the jets around the hero; these probes want the pure seeded jet
    # background, so neutralize it (local_jet already 0 in the baked warm).
    p.jets.hero_bracket_north = 0.0
    p.jets.hero_bracket_south = 0.0
    for key, val in storms_over.items():
        setattr(p.storms, key, val)
    assert p.solver.type == SolverType.VORTICITY
    return p


def _dev0_omega(p: PlanetParams, gpu):
    """BYTE-EXACT asserts on this texture are a documented CARVE-OUT from the
    vorticity-tolerance rule: the dev-0 omega state is a single analytic
    per-pixel dispatch (omega_init: jets + vortex stamps + f, confined) —
    read back BEFORE any SOR or advection pass touches it. Comparisons must
    stay same-process (shared program cache); like the other byte-exact GPU
    classes this may join the RTX session-flake list — rerun once before
    investigating."""
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    q = np.asarray(sim.gpu.read_texture(sim.solver._omega_state.cur))
    return sim, np.squeeze(q).astype(np.float64).copy()


def test_hero_flow_aspect_lever(gpu):
    """storms.hero_flow_aspect: K=1 is deterministic (byte-equal reruns —
    determinism, NOT a no-op claim: the cross-commit two-tier capture is the
    no-op evidence, see the S1 commit); K=2 changes the flow; and it composes
    with the shape lobes and the taper wedge (which ride the K-widened metric
    by design — the deforms live in the ring's own frame)."""
    def omega(k: float, shape: float = 0.0, taper: float = 0.0) -> np.ndarray:
        p = _solo_warm_params(
            hero_flow_aspect=k, hero_shape=shape, hero_taper=taper)
        return _dev0_omega(p, gpu)[1]

    base = omega(1.0)
    np.testing.assert_array_equal(base, omega(1.0))     # determinism at K=1
    wide = omega(2.0)
    assert np.abs(wide - base).max() > 1e-2, "K=2 did not change the flow"
    with_lobes = omega(2.0, shape=1.0)
    assert np.abs(with_lobes - wide).max() > 1e-3, "lobes inert on top of K"
    with_taper = omega(2.0, taper=1.0)
    assert np.abs(with_taper - wide).max() > 1e-3, "taper inert on top of K"


def _trough_radius(prof: np.ndarray, center: int, lo: int, hi: int) -> float:
    """Distance (in samples) from `center` to the minimum of `prof` within
    [center+lo, center+hi], refined to sub-sample by parabolic interpolation
    (the NS ring trough sits ~5 samples out even at 1024 — argmin quantization
    alone would eat the tolerance)."""
    seg = prof[center + lo: center + hi]
    i = int(np.argmin(seg))
    if 0 < i < len(seg) - 1:
        a, b, c = seg[i - 1], seg[i], seg[i + 1]
        denom = a - 2.0 * b + c
        if denom > 1e-12:
            i += 0.5 * (a - c) / denom
    return float(abs(i + lo))


def test_hero_flow_aspect_widens_flow_ew_only(gpu):
    """K widens the vorticity ring's EW extent by ~K, leaves its NS extent
    unchanged, and preserves the ring+skirt NET circulation via the
    CPU-computed spherical renorm (sim/flow_renorm.py — plain tangent-plane
    1/K would leave a ~16% net deficit at K=2, the taper's measured
    planet-wide band-shift class). The circulation probe uses the K-vs-1
    DIFFERENCE field, which cancels the (identical) jets and f exactly."""
    K = 2.0

    def scene(k: float):
        p = _solo_warm_params(hero_flow_aspect=k)
        p.sim.resolution = 1024
        return _dev0_omega(p, gpu)

    sim1, q1 = scene(1.0)
    _, qk = scene(K)
    hero = sim1.vortices.heroes()[0]
    h, w = q1.shape
    row = int(round((0.5 - hero.lat / np.pi) * h - 0.5))
    col = int(round((hero.lon + np.pi) / (2.0 * np.pi) * w - 0.5))

    # Ring trough radius along each axis (min of the zonal anomaly), searched
    # in the vortex's OWN core polarity (core omega = -sign(strength), the
    # chirality convention pinned by fix/vortex-chirality 4b60fa6/module
    # docstring): the Laplacian-of-Gaussian radial shape has one sign at the
    # core (q~0) and the opposite sign at the ring shoulder (q~1.22) — which
    # extremum "argmin" lands on depends on that polarity, not on the ring's
    # physical location. This probe's _solo_warm_params pin (hero_latitude
    # -21, predating the chirality fix) sits at an ambient latitude whose
    # _ambient_sign is UNCHANGED by the fix (measured: -1.0 identically on
    # both sides of 4b60fa6, jet-independent — the new local_jet_speed bake
    # term is a per-row constant that cancels exactly under the row-mean
    # subtraction below, confirmed jet-on vs jet-off byte-identical here) —
    # so `strength` itself flips sign (-0.0855 pre-fix -> +0.0855 post-fix,
    # co-rotation was the bug being fixed), and unoriented argmin silently
    # started grabbing the near-core dip (ew_r 15px) instead of the outer
    # ring shoulder (32px) it always used to find. Orienting by core polarity
    # reproduces the pre-fix master measurement bit-for-bit (32/65 -> ratio
    # 2.0312, both jet on and off, both hero_latitude -21 and the new -22
    # bake) — this is a search-heuristic fix, not a relaxed tolerance.
    orient = -1.0 if hero.strength >= 0.0 else 1.0

    dlon_px = 2.0 * np.pi / w
    dlat_px = np.pi / h
    ew_r, ns_r = {}, {}
    for k, q in ((1.0, q1), (K, qk)):
        anom = (q - q.mean(axis=1, keepdims=True)) * orient
        lo_e = 2
        hi_e = int(1.3 * hero.aspect * k * hero.r_core / np.cos(hero.lat)
                   / dlon_px)
        east = _trough_radius(anom[row], col, lo_e, hi_e)
        west = _trough_radius(anom[row][::-1], w - 1 - col, lo_e, hi_e)
        ew_r[k] = 0.5 * (east + west)
        hi_n = int(1.3 * hero.r_core / dlat_px)
        north = _trough_radius(anom[:, col][::-1], h - 1 - row, 1, hi_n)
        south = _trough_radius(anom[:, col], row, 1, hi_n)
        ns_r[k] = 0.5 * (north + south)

    ew_ratio = ew_r[K] / ew_r[1.0]
    ns_ratio = ns_r[K] / ns_r[1.0]
    assert abs(ew_ratio / K - 1.0) < 0.06, (
        f"EW ring trough scaled x{ew_ratio:.3f}, expected ~x{K}"
    )
    assert abs(ns_ratio - 1.0) < 0.05, (
        f"NS ring trough moved x{ns_ratio:.3f}, expected unchanged"
    )

    # Net circulation invariance: sum of the (K - 1) difference over the hero
    # neighborhood, cos-lat weighted, relative to the ring's own mass. A 1/K
    # renorm bug reads ~0.02 on this statistic; the spherical renorm ~1e-4.
    # The box must COVER the K-widened skirt's EW support (outer edge at
    # widened-q 2.4, i.e. ~2.4*aspect*K*r_core of longitude at this latitude)
    # or the truncation itself reads as a ~0.03 circulation error — a fixed
    # 0.65 rad half-width did exactly that when warm's bake moved hero_aspect
    # 2.2 -> 2.9 (support 0.71 rad), while the renorm was in fact at 1e-4.
    lat_g = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    cosw = np.cos(lat_g)[:, None]
    rows = slice(max(row - int(0.16 / dlat_px), 0), row + int(0.16 / dlat_px))
    cs = int(2.4 * hero.aspect * K * hero.r_core / np.cos(hero.lat) * 1.15
             / dlon_px)
    cols = slice(max(col - cs, 0), min(col + cs, w))
    box = (rows, cols)
    net_diff = ((qk - q1) * cosw)[box].sum()
    anom1 = q1 - q1.mean(axis=1, keepdims=True)
    ring_mass = (np.abs(anom1) * cosw)[box].sum()
    # 3e-3 floor (PR-43 test review): the full 1/K-instead-of-spherical bug
    # reads ~0.02 on this statistic and healthy ~1e-4 — the old 0.01 floor
    # left only 2x separation, letting a half-broken renorm through.
    assert abs(net_diff) / ring_mass < 3e-3, (
        f"net circulation moved by {abs(net_diff) / ring_mass:.4f} of the "
        f"ring mass — the u_hero_flow_renorm compensation is off"
    )


def test_hero_taper_preserves_net_circulation(gpu):
    """The taper's omega-side compensation (`tcomp *= 1/(1 - 0.105*t*e)`) is
    the constant with planet-wide blast radius: this exact mechanism shipped
    computed-but-never-applied once (ca76f00), and its ~9% deficit class
    measurably shifted bands 25+ degrees through the global Poisson solve.
    The linearization is validated against wedge quadrature GL-free
    (test_hero_shape_constants); THIS probe proves the shader actually
    applies it: the taper-vs-base dev-0 difference field (jets and f cancel
    exactly) must carry no net circulation beyond the documented +0.8%
    over-compensation residual at warm defaults (~0.2% of ring mass; floor
    0.01 with margin)."""
    def scene(taper: float):
        p = _solo_warm_params(hero_taper=taper, hero_emergence=0.9)
        p.sim.resolution = 1024
        return _dev0_omega(p, gpu)

    sim0, q0 = scene(0.0)
    _, qt = scene(1.0)
    hero = sim0.vortices.heroes()[0]
    h, w = q0.shape
    row = int(round((0.5 - hero.lat / np.pi) * h - 0.5))
    col = int(round((hero.lon + np.pi) / (2.0 * np.pi) * w - 0.5))
    dlat_px = np.pi / h
    dlon_px = 2.0 * np.pi / w
    lat_g = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    cosw = np.cos(lat_g)[:, None]
    rows = slice(max(row - int(0.16 / dlat_px), 0), row + int(0.16 / dlat_px))
    # The wedge only pulls the boundary IN, so the K=1 support bound suffices.
    cs = int(2.4 * hero.aspect * hero.r_core / np.cos(hero.lat) * 1.15 / dlon_px)
    cols = slice(max(col - cs, 0), min(col + cs, w))
    box = (rows, cols)
    net_diff = ((qt - q0) * cosw)[box].sum()
    anom0 = q0 - q0.mean(axis=1, keepdims=True)
    ring_mass = (np.abs(anom0) * cosw)[box].sum()
    assert abs(net_diff) / ring_mass < 0.01, (
        f"net circulation moved by {abs(net_diff) / ring_mass:.4f} of the "
        f"ring mass under hero_taper=1 — the 0.105 omega compensation is "
        f"stale or unapplied (the ca76f00 class)"
    )


def test_hero_flow_aspect_flow_stays_anchored_at_hi(gpu):
    """K=2.5 (the pfield hi bound) is where the UNWIDENED anchor basin has
    its thinnest coverage of the widened ring (wings at anatomy-q 2.6 vs the
    window's 1.6-2.8 fade — partial boost ~0.17). The capture basin was
    deliberately NOT widened with K (widening it drops the ~44x nudge damping
    onto the whole wake wedge); this test is the standing evidence that the
    unwidened basin still holds the widened OMEGA ring across the slider
    range: deep-ring latitude centroid glued to the registry (the historical
    wander bug was ~0.2 rad ~ 11 deg) and the ring's longitude span
    bracketing the registry. Deliberately probes OMEGA, not the dye: the
    S1 investigation measured the flow anchored at every K (lat centroid
    +0.1 deg at K=2) while the RED FILL dilutes with K (T3 at registry,
    K=1 -> K=2: 0.62 -> 0.35 at 2048/dev700, 0.44 -> 0.10 at 512/dev300 —
    grid-diffusion + the anatomy-metric flush stripping dye on every EW
    transit of the widened eddy). Dye containment vs K is an S2 calibration
    item and caps the PRESET bake (the plain anchor test above guards that
    at whatever K warm ships); it is not this lever's kernel contract."""
    p = _solo_warm_params(hero_flow_aspect=2.5, hero_emergence=1.0)
    p.sim.dev_steps = 300
    sim, _ = _sim_and_tracers(p, gpu)
    q = np.squeeze(np.asarray(
        sim.gpu.read_texture(sim.solver._omega_state.cur))).astype(np.float64)
    hero = sim.vortices.heroes()[0]
    h, w = q.shape
    row = int((0.5 - hero.lat / np.pi) * h)
    col = int((hero.lon + np.pi) / (2.0 * np.pi) * w)
    anom = q - q.mean(axis=1, keepdims=True)
    rr = int(0.15 / (np.pi / h))
    cc = int(0.75 / (2.0 * np.pi / w))
    box = anom[row - rr: row + rr, col - cc: col + cc]
    deep = box < 0.5 * box.min()
    ys, xs = np.where(deep)
    assert deep.sum() > 50, "no deep vorticity ring near the registry at all"
    wgt = -box[deep]
    lat_off = ((ys - rr) * wgt).sum() / wgt.sum() * (np.pi / h)
    assert abs(np.degrees(lat_off)) < 1.5, (
        f"deep-ring latitude centroid {np.degrees(lat_off):+.2f} deg off the "
        f"registry at hero_flow_aspect 2.5 — the widened ring escaped the "
        f"anchor basin (measured +0.44 deg when healthy)"
    )
    assert xs.min() < cc < xs.max(), (
        "the deep ring's longitude span no longer brackets the registry — "
        "the storm slid along the band"
    )

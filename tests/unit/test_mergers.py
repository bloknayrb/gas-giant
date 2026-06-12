"""Vortex mergers: criterion, conservation, protections, determinism.

CPU-only — the registry/profile machinery is GPU-free."""

from __future__ import annotations

import numpy as np

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.advance import advance_registry
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import LatProfiles, build_profiles
from gasgiant.sim.solver import compute_dt
from gasgiant.sim.vortices import (
    _V_PEAK_COEF,
    KIND_BARGE,
    KIND_DEBRIS,
    KIND_HERO,
    KIND_KH,
    KIND_OVAL,
    KIND_PEARL,
    KIND_POLAR,
    MAX_VORTICES,
    MERGE_COOLDOWN,
    MERGE_DEBRIS_LIFETIME,
    MERGE_DEBRIS_RAMP,
    MERGE_MAX_R,
    MERGE_V_MAX,
    Vortex,
    VortexRegistry,
    _seed_convergent_pairs,
    generate_vortices,
    resolve_mergers,
)


def _synth_profiles(shear: float = 1.0) -> LatProfiles:
    """u = shear * lat: monotone differential drift everywhere."""
    n = 512
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, n)
    u = shear * lat
    z = np.zeros(n)
    return LatProfiles(lat=lat, u=u, psi=z, shear_norm=z, belt_mask=z,
                       t0_stamp=z, t1_stamp=z, max_speed=float(np.abs(u).max()))


def _storms(rate: float = 1.0):
    p = PlanetParams()
    p.storms.merge_rate = rate
    return p.storms


def _converging_pair(r: float = 0.03):
    """Same-sign ovals: b sits poleward (faster drift, u = lat) and east-behind
    so the gap closes."""
    a = Vortex(0.30, 0.00, r, 0.012, KIND_OVAL, tint=0.1, brightness=0.2)
    b = Vortex(0.33, -0.05, r, 0.010, KIND_OVAL, tint=0.3, brightness=0.3)
    return a, b


def _fields(reg):
    return [
        (v.lat, v.lon, v.r_core, v.strength, v.kind, v.tint, v.brightness,
         v.wake_dir, v.cooldown)
        for v in reg.vortices
    ]


def _product(reg):
    """The merge product: a live (non-transient) oval with a cooldown."""
    out = [v for v in reg.vortices
           if v.kind == KIND_OVAL and abs(v.strength) > 1e-6 and v.cooldown > 0]
    assert len(out) == 1
    return out[0]


def test_converging_pair_merges():
    a, b = _converging_pair()
    reg = VortexRegistry([a, b])
    resolved = resolve_mergers(reg, _synth_profiles(), _storms())
    assert len(resolved) == 1
    assert len(reg.vortices) == 2  # product + debris collar
    prod = _product(reg)
    assert prod.kind == KIND_OVAL
    assert prod.cooldown == MERGE_COOLDOWN
    debris = [v for v in reg.vortices if v.kind == KIND_DEBRIS]
    assert len(debris) == 1
    assert debris[0].strength == 0.0  # no psi contribution
    assert debris[0].ttl == MERGE_DEBRIS_LIFETIME
    assert debris[0].r_core == prod.r_core


def test_conservation_uncapped():
    a, b = _converging_pair()
    reg = VortexRegistry([a, b])
    resolve_mergers(reg, _synth_profiles(), _storms())
    prod = _product(reg)
    # Peak-velocity conservation: S*r is the invariant.
    np.testing.assert_allclose(
        abs(prod.strength) * prod.r_core,
        abs(a.strength) * a.r_core + abs(b.strength) * b.r_core,
    )
    np.testing.assert_allclose(prod.r_core, np.hypot(a.r_core, b.r_core))
    w1 = abs(a.strength) * a.r_core**2
    w2 = abs(b.strength) * b.r_core**2
    np.testing.assert_allclose(prod.lat, (w1 * a.lat + w2 * b.lat) / (w1 + w2))
    np.testing.assert_allclose(prod.tint, (w1 * a.tint + w2 * b.tint) / (w1 + w2))


def test_seam_lon_interp():
    a = Vortex(0.30, 3.12, 0.03, 0.012, KIND_OVAL)
    b = Vortex(0.33, -3.12, 0.03, 0.010, KIND_OVAL)
    reg = VortexRegistry([a, b])
    resolve_mergers(reg, _synth_profiles(-1.0), _storms())  # shear sign -> converging
    # Product near the seam, NOT near lon 0 (naive mean of +3.12/-3.12).
    assert abs(_product(reg).lon) > 3.0


def test_capped_branch_velocity_bounded():
    a = Vortex(0.30, 0.00, 0.07, 0.030, KIND_OVAL)
    b = Vortex(0.33, -0.05, 0.07, 0.030, KIND_OVAL)
    reg = VortexRegistry([a, b])
    resolve_mergers(reg, _synth_profiles(), _storms())
    prod = _product(reg)
    assert prod.r_core == MERGE_MAX_R  # sqrt(2)*0.07 > cap
    assert _V_PEAK_COEF * abs(prod.strength) / prod.r_core <= MERGE_V_MAX + 1e-12


def test_chain_merges_respect_velocity_cap():
    profiles = _synth_profiles()
    storms = _storms()
    survivor = Vortex(0.30, 0.00, 0.055, 0.022, KIND_OVAL, cooldown=1)
    reg = VortexRegistry([survivor])
    for _ in range(6):
        prod = _product(reg)
        prod.cooldown = 1  # bypass hysteresis: this test is about the caps
        challenger = Vortex(
            prod.lat + 0.02, prod.lon - 0.04, 0.055,
            0.022 if prod.strength > 0 else -0.022, KIND_OVAL,
        )
        reg.vortices.append(challenger)
        resolved = resolve_mergers(reg, profiles, storms)
        assert len(resolved) == 1
        prod = _product(reg)
        v_peak = _V_PEAK_COEF * abs(prod.strength) / prod.r_core
        assert v_peak <= MERGE_V_MAX + 1e-12
        assert prod.r_core <= MERGE_MAX_R


def test_opposite_sign_never_merges():
    a, b = _converging_pair()
    b.strength = -b.strength
    reg = VortexRegistry([a, b])
    assert resolve_mergers(reg, _synth_profiles(), _storms()) == []
    assert len(reg.vortices) == 2


def test_receding_pair_never_merges():
    a, b = _converging_pair()
    b.lon = +0.05  # east-ahead of the faster-drifting b: gap opens
    reg = VortexRegistry([a, b])
    assert resolve_mergers(reg, _synth_profiles(), _storms()) == []


def test_exact_same_lat_never_merges():
    """The strict converging gate (closing > 0.0): identical latitudes give a
    bit-identical drift rate and a closing rate of exactly 0.0. This is the
    structural protection for pearls and pre-sheared twins."""
    a = Vortex(0.30, 0.00, 0.02, 0.008, KIND_OVAL)
    twin = Vortex(0.30, 0.05, 0.026, 0.0032, KIND_OVAL)  # inside capture
    reg = VortexRegistry([a, twin])
    for _ in range(500):
        advance_registry(reg, _synth_profiles(), 0.002, 0, storms=_storms())
    assert len(reg.vortices) == 2


def test_pearls_survive_dev_run():
    p = PlanetParams(seed=42)
    p.storms.merge_rate = 1.0
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    dt = compute_dt(p.sim.resolution, p.sim.dt_scale, profiles.max_speed)
    reg = generate_vortices(p.seed, bands, profiles, p.storms, p.poles,
                            dt=dt, dev_steps=500)
    pearls_before = sum(1 for v in reg.vortices if v.kind == KIND_PEARL)
    assert pearls_before > 0
    for i in range(500):
        advance_registry(reg, profiles, dt, i, storms=p.storms)
    assert sum(1 for v in reg.vortices if v.kind == KIND_PEARL) == pearls_before


def test_hero_absorbs_oval():
    hero = Vortex(0.30, 0.00, 0.10, 0.045, KIND_HERO, tint=0.9, brightness=0.05,
                  wake_dir=1.0)
    before = (hero.lat, hero.lon, hero.r_core, hero.strength, hero.tint,
              hero.brightness, hero.wake_dir)
    oval = Vortex(0.33, -0.08, 0.03, 0.012, KIND_OVAL)
    reg = VortexRegistry([hero, oval])
    resolved = resolve_mergers(reg, _synth_profiles(), _storms())
    assert len(resolved) == 1 and resolved[0][2] is None
    assert reg.vortices[0] is hero
    assert (hero.lat, hero.lon, hero.r_core, hero.strength, hero.tint,
            hero.brightness, hero.wake_dir) == before
    # Shredded-victim debris at the victim's position and size.
    debris = [v for v in reg.vortices if v.kind == KIND_DEBRIS]
    assert len(debris) == 1
    assert (debris[0].lat, debris[0].lon, debris[0].r_core) == (
        oval.lat, oval.lon, oval.r_core)


def test_hero_hero_never_merges():
    h1 = Vortex(0.30, 0.00, 0.10, 0.045, KIND_HERO)
    h2 = Vortex(0.33, -0.08, 0.10, 0.045, KIND_HERO)
    reg = VortexRegistry([h1, h2])
    assert resolve_mergers(reg, _synth_profiles(), _storms()) == []


def test_pearl_cross_kind_excluded():
    """A zone oval converging onto the pearl latitude must NOT eat a pearl —
    the string's evenness is a designed formation."""
    pearl = Vortex(0.30, 0.00, 0.02, 0.008, KIND_PEARL)
    oval = Vortex(0.33, -0.04, 0.03, 0.012, KIND_OVAL)
    reg = VortexRegistry([pearl, oval])
    assert resolve_mergers(reg, _synth_profiles(), _storms()) == []


def test_whitelist_inert_kinds():
    profiles = _synth_profiles()
    for kind in (KIND_BARGE, KIND_KH, KIND_POLAR):
        a = Vortex(0.30, 0.00, 0.03, 0.012, kind)
        b = Vortex(0.33, -0.05, 0.03, 0.012, kind)
        reg = VortexRegistry([a, b])
        assert resolve_mergers(reg, profiles, _storms()) == [], kind
    # Zero-strength transients (outbreaks, debris) are excluded by |s| > 1e-6.
    a = Vortex(0.30, 0.00, 0.03, 0.0, KIND_OVAL)
    b = Vortex(0.33, -0.05, 0.03, 0.0, KIND_OVAL)
    assert resolve_mergers(VortexRegistry([a, b]), profiles, _storms()) == []


def test_kind_constants_distinct():
    from gasgiant.sim.events import KIND_OUTBREAK

    kinds = [KIND_OVAL, KIND_HERO, KIND_BARGE, KIND_PEARL, KIND_KH, KIND_POLAR,
             KIND_OUTBREAK, KIND_DEBRIS]
    assert len(set(kinds)) == len(kinds)


def test_debris_fades_in_decays_and_expires():
    a, b = _converging_pair()
    reg = VortexRegistry([a, b])
    profiles = _synth_profiles()
    resolve_mergers(reg, profiles, _storms())
    debris = next(v for v in reg.vortices if v.kind == KIND_DEBRIS)
    assert debris.brightness == 0.0  # spawns dark, ramps in (no target pop)

    seen = []
    for _ in range(MERGE_DEBRIS_LIFETIME + 1):
        resolve_mergers(reg, profiles, _storms())
        if debris in reg.vortices:
            seen.append(debris.brightness)
    assert all(v.kind != KIND_DEBRIS for v in reg.vortices)  # expired + removed
    peak = max(seen)
    assert peak > 0.5  # 0.9 * merge_debris(1.0) * stamp_contrast(1.0), ramped
    assert seen[: MERGE_DEBRIS_RAMP - 1] == sorted(seen[: MERGE_DEBRIS_RAMP - 1])
    assert seen[-1] < 0.1 * peak  # decayed to ~nothing before removal


def test_debris_skipped_near_exchange_band():
    lat0 = np.deg2rad(60.0)
    a = Vortex(lat0, 0.00, 0.03, 0.012, KIND_OVAL)
    b = Vortex(lat0 + 0.03, -0.05, 0.03, 0.010, KIND_OVAL)
    reg = VortexRegistry([a, b])
    resolved = resolve_mergers(reg, _synth_profiles(), _storms())
    assert len(resolved) == 1  # the merge itself happens
    assert all(v.kind != KIND_DEBRIS for v in reg.vortices)  # collar suppressed


def test_debris_disabled_by_param():
    a, b = _converging_pair()
    reg = VortexRegistry([a, b])
    storms = _storms()
    storms.merge_debris = 0.0
    resolved = resolve_mergers(reg, _synth_profiles(), storms)
    assert len(resolved) == 1
    assert all(v.kind != KIND_DEBRIS for v in reg.vortices)


def test_greedy_one_merge_per_step_and_cooldown():
    a = Vortex(0.300, 0.000, 0.03, 0.012, KIND_OVAL)
    b = Vortex(0.330, -0.040, 0.03, 0.012, KIND_OVAL)
    c = Vortex(0.355, -0.085, 0.03, 0.012, KIND_OVAL)
    reg = VortexRegistry([a, b, c])
    profiles = _synth_profiles()
    resolved = resolve_mergers(reg, profiles, _storms())
    assert len(resolved) == 1  # the cluster does not collapse in one step
    assert len([v for v in reg.vortices if abs(v.strength) > 1e-6]) == 2
    # The product carries a cooldown: no second merge until it expires.
    merges = 0
    for _ in range(MERGE_COOLDOWN - 1):
        merges += len(resolve_mergers(reg, profiles, _storms()))
    assert merges == 0


def test_rate_zero_is_bare_drift():
    p = PlanetParams(seed=7)
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    reg_a = generate_vortices(p.seed, bands, profiles, p.storms, p.poles)
    reg_b = generate_vortices(p.seed, bands, profiles, p.storms, p.poles)
    assert p.storms.merge_rate == 0.0
    for i in range(50):
        advance_registry(reg_a, profiles, 0.002, i, storms=p.storms)
        reg_b.drift(profiles, 0.002)
    assert _fields(reg_a) == _fields(reg_b)


def test_determinism():
    def run():
        p = PlanetParams(seed=11)
        p.storms.merge_rate = 0.7
        bands = generate_bands(p.seed, p.bands)
        profiles = build_profiles(p.seed, bands, p.bands, p.jets)
        dt = compute_dt(p.sim.resolution, p.sim.dt_scale, profiles.max_speed)
        reg = generate_vortices(p.seed, bands, profiles, p.storms, p.poles,
                                dt=dt, dev_steps=500)
        for i in range(300):
            advance_registry(reg, profiles, dt, i, storms=p.storms)
        return _fields(reg)

    assert run() == run()


def test_seeded_pairs_actually_merge():
    """THE regression guard for the closure arithmetic: kinematically placed
    pairs must produce real mergers within a default-length dev run."""
    total = 0
    for seed in (1, 42, 9120):
        p = PlanetParams(seed=seed)
        p.storms.merge_rate = 0.7
        bands = generate_bands(p.seed, p.bands)
        profiles = build_profiles(p.seed, bands, p.bands, p.jets)
        dt = compute_dt(p.sim.resolution, p.sim.dt_scale, profiles.max_speed)
        reg = generate_vortices(p.seed, bands, profiles, p.storms, p.poles,
                                dt=dt, dev_steps=500)
        merges = 0
        for _ in range(500):  # mirrors advance_registry (events=None)
            reg.drift(profiles, dt)
            merges += len(resolve_mergers(reg, profiles, p.storms))
        assert merges >= 1, f"seed {seed}: no seeded pair merged"
        total += merges
    assert total >= 3


def test_seeding_off_at_rate_zero_and_without_dt():
    p = PlanetParams(seed=5)
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    base = generate_vortices(p.seed, bands, profiles, p.storms, p.poles)
    with_dt = generate_vortices(p.seed, bands, profiles, p.storms, p.poles,
                                dt=0.002, dev_steps=500)
    assert _fields(VortexRegistry(base.vortices)) == _fields(VortexRegistry(with_dt.vortices))
    p.storms.merge_rate = 0.7
    no_dt = generate_vortices(p.seed, bands, profiles, p.storms, p.poles)
    assert len(no_dt.vortices) == len(base.vortices)


def test_seeded_pair_cap_is_atomic():
    reg = VortexRegistry([
        Vortex(0.3, float(lon), 0.02, 0.008, KIND_OVAL)
        for lon in np.linspace(-3.0, 3.0, MAX_VORTICES)
    ])
    rng = np.random.default_rng(0)
    _seed_convergent_pairs(
        reg, rng, [(0.3, 0.2)] * 8, _synth_profiles(), 1.0, 0.002, 500
    )
    assert len(reg.vortices) <= MAX_VORTICES


def test_merge_params_are_restart_tier():
    a = PlanetParams()
    b = PlanetParams()
    b.storms.merge_rate = 0.5
    assert diff_tiers(a, b) == {Tier.RESTART}
    c = PlanetParams()
    c.storms.merge_debris = 0.5
    assert diff_tiers(a, c) == {Tier.RESTART}

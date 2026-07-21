"""detail.hero_wake_billows param/plumbing invariants (no GL).

The GPU behavior tests (tests/gpu/test_hero_wake_billows.py) are deselected from
the always-blocking no-GPU tier; the invalidation tier and the snapshot 8-tuple
premise gate every PR from here. (fx=True metadata coverage -- predicate,
tripwire, no-rand, dispatch cross-ref -- is owned by test_detail_fx_metadata.py's
derived lists; the billows freq companion staying OUT of that list is asserted
there too.)
"""
from __future__ import annotations

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.engine.snapshot import hero_centers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import generate_vortices


def test_hero_wake_billows_is_post_tier():
    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.detail.hero_wake_billows = 0.7
    assert diff_tiers(a, b) == {Tier.POST}


def test_hero_wake_billow_freq_is_post_tier():
    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.detail.hero_wake_billow_freq = 1.4
    assert diff_tiers(a, b) == {Tier.POST}


def test_snapshot_wake_frame_is_the_shared_8_tuple():
    """Billows consumes the SAME registry wake frame the braid does -- no new
    snapshot fields. The 8-tuple (x, y, z, r_core, spin, aspect, wake_dir,
    wake_lat_off) is unchanged (the wiring reuses wdirs/wlats/wbelt), so this
    pins len == 8 exactly as the braid test does."""
    p = PlanetParams(seed=42)
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    reg = generate_vortices(p.seed, bands, profiles, p.storms, p.poles)
    heroes = reg.heroes()
    assert heroes, "seed 42 must seed a hero"
    tuples = hero_centers(reg)
    assert len(tuples) == len(heroes)
    for t, v in zip(tuples, heroes, strict=True):
        assert len(t) == 8
        assert t[3] == v.r_core
        assert t[6] == v.wake_dir
        assert t[7] == v.wake_lat_off

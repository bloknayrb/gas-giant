"""detail.hero_wake_braid param/plumbing invariants (no GL).

The GPU behavior tests (tests/gpu/test_hero_wake_braid.py) are deselected
from the always-blocking no-GPU tier; the invalidation tier and the wake-frame
plumbing gate every PR from here. (fx=True metadata coverage — predicate,
tripwire, no-rand — is owned by test_detail_fx_metadata.py's derived lists.)
"""
from __future__ import annotations

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.engine.snapshot import hero_centers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import generate_vortices


def test_hero_wake_braid_is_post_tier():
    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.detail.hero_wake_braid = 0.7
    assert diff_tiers(a, b) == {Tier.POST}


def test_hero_centers_carries_the_wake_frame():
    """The braid's wake geometry comes from the sim registry through the
    snapshot 8-tuple (x, y, z, r_core, spin, aspect, wake_dir, wake_lat_off)
    — render and sim must agree on where the wake is, so the tuple mirrors
    the registry fields verbatim."""
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
        assert t[5] == v.aspect
        assert t[6] == v.wake_dir
        assert t[7] == v.wake_lat_off
        assert t[6] in (-1.0, 1.0)  # heroes always carry a trail direction

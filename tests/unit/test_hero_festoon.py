"""waves.festoon_hero_strength (FESTOON2) CPU pieces + the hero-relative
accent-longitude rule.

CPU-only: the latitude selector (sim/profiles.py::select_hero_festoon_latitude)
and the accent placement rule (sim/vortices.py::_add_accent_ovals) are registry/
profile data; the GPU behavior lives in tests/gpu/test_festoon_hero.py.
"""
from __future__ import annotations

import numpy as np

from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import (
    build_profiles,
    select_hero_festoon_latitude,
    select_wave_latitudes,
)
from gasgiant.sim.vortices import KIND_OVAL, generate_vortices


def _warm_scene(**storms_update):
    p = load_factory_preset("gas_giant_warm")
    if storms_update:
        p.storms = p.storms.model_copy(update=storms_update)
    bands = generate_bands(p.seed, p.bands)
    prof = build_profiles(p.seed, bands, p.bands, p.jets)
    reg = generate_vortices(p.seed, bands, prof, p.storms, p.poles)
    return p, bands, prof, reg


# ------------------------------------------------ select_hero_festoon_latitude

def test_selects_nearest_interior_edge_on_warm():
    """warm's hero straddles the belt edge at ~-19.4 deg: the selector must
    return exactly the interior edge nearest the hero, within its 0.15-rad
    reach, and never the primary (+7-deg-side) festoon edge."""
    _, bands, prof, reg = _warm_scene()
    hero = reg.heroes()[0]
    fest, _rib = select_wave_latitudes(bands, prof)
    lat = select_hero_festoon_latitude(bands, hero.lat, fest)
    assert lat is not None
    interior = bands.edges[1:-1].astype(np.float64)
    nearest = float(interior[int(np.argmin(np.abs(interior - hero.lat)))])
    assert lat == nearest
    assert abs(lat - hero.lat) <= 0.15
    assert abs(lat - fest) > 1e-6


def test_none_when_no_edge_within_reach():
    """A hero far from every interior edge gets no train (a train rooted away
    from the storm is just a second equatorial comb)."""
    _, bands, _prof, _reg = _warm_scene()
    interior = bands.edges[1:-1].astype(np.float64)
    # Probe latitude 0.2 rad past the poleward-most interior edge: by
    # construction >0.15 from every edge.
    probe = float(interior.max()) + 0.2
    assert select_hero_festoon_latitude(bands, probe, 0.12) is None


def test_never_double_trains_the_primary_edge():
    """If the nearest edge IS the primary festoon edge, the selector must
    decline rather than stack two trains on one latitude."""
    _, bands, _prof, _reg = _warm_scene()
    interior = bands.edges[1:-1].astype(np.float64)
    edge = float(interior[0])
    assert select_hero_festoon_latitude(bands, edge, edge) is None


# ------------------------------------------------ hero-relative accent placement

def test_pinned_latitude_accent_roots_downstream_of_hero():
    """accent_latitude pinned + accent_longitude None + hero present: the
    accent must sit a seeded 0.3-0.55 rad DOWNSTREAM of the hero (the
    Oval-BA-passing-south recipe) — an unpinned longitude would put it out of
    any hero-framed view ~90% of the time."""
    p, _bands, _prof, reg = _warm_scene(
        accent_count=1, accent_latitude=-29.0, accent_tint=0.77
    )
    hero = reg.heroes()[0]
    accents = [
        v for v in reg.vortices
        if v.kind == KIND_OVAL and v.tint == p.storms.accent_tint
    ]
    assert len(accents) == 1
    acc = accents[0]
    assert np.isclose(acc.lat, np.deg2rad(-29.0))
    d = (acc.lon - hero.lon + np.pi) % (2.0 * np.pi) - np.pi
    along = d * hero.wake_dir
    assert 0.3 <= along <= 0.55, (
        f"accent sits {along:+.2f} rad along the wake axis — not in the "
        "authored 0.3-0.55 downstream window"
    )


def test_explicit_longitude_pin_still_wins():
    """accent_longitude explicit overrides the hero-relative rule (the rule
    only fills the None default)."""
    p, _bands, _prof, reg = _warm_scene(
        accent_count=1, accent_latitude=-29.0, accent_longitude=1.0,
        accent_tint=0.77,
    )
    acc = [
        v for v in reg.vortices
        if v.kind == KIND_OVAL and v.tint == p.storms.accent_tint
    ][0]
    # generate_vortices ran without dt/dev_steps here, so the drift-compensated
    # pin reduces to the requested longitude (accent_longitude is in DEGREES,
    # like every user-facing lat/lon param).
    assert np.isclose(acc.lon, np.deg2rad(1.0))


def test_seeded_zone_path_is_deterministic():
    """accent_latitude None (the seeded-zone path — e.g. neptune's Scooter)
    stays deterministic; the rule's appended draw must not perturb it
    run-to-run."""
    _, _, _, reg1 = _warm_scene(accent_count=1, accent_tint=0.77)
    _, _, _, reg2 = _warm_scene(accent_count=1, accent_tint=0.77)
    a1 = [v for v in reg1.vortices if v.kind == KIND_OVAL and v.tint == 0.77][0]
    a2 = [v for v in reg2.vortices if v.kind == KIND_OVAL and v.tint == 0.77][0]
    assert a1.lat == a2.lat and a1.lon == a2.lon

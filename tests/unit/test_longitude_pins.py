"""T1 longitude pins: hero_longitude / accent_longitude / outbreak_longitude.

The pins name the RENDERED (end-of-dev-run) longitude; the generator
inverse-compensates the closed-form zonal drift so the storm lands where you
asked. Default (all None) must be byte-identical to today, and randomize must
never touch them (no ``rand`` metadata)."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, StormsParams, Tier
from gasgiant.params.randomize import randomize
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.solver import compute_dt
from gasgiant.sim.vortices import KIND_HERO, KIND_OVAL, generate_vortices

# ---------------------------------------------------------------- helpers


def _profiles(seed: int, p: PlanetParams):
    bands = generate_bands(seed, p.bands)
    profiles = build_profiles(seed, bands, p.bands, p.jets)
    return bands, profiles


def _dt(p: PlanetParams, profiles) -> float:
    return compute_dt(p.sim.resolution, p.sim.dt_scale, profiles.max_speed)


def _hero(reg):
    heroes = [v for v in reg.vortices if v.kind == KIND_HERO]
    assert heroes, "no KIND_HERO vortex found"
    return heroes[0]


# ---------------------------------------------------------------- byte-identity


def test_default_none_longitude_identical():
    """All three pins None ⇒ registry field-by-field identical to default."""
    p_default = PlanetParams()
    p_explicit = PlanetParams()
    p_explicit.storms.hero_longitude = None
    p_explicit.storms.accent_longitude = None
    p_explicit.storms.outbreak_longitude = None

    assert p_default.storms.hero_longitude is None
    assert p_default.storms.accent_longitude is None
    assert p_default.storms.outbreak_longitude is None

    seed = 7
    bands, prof = _profiles(seed, p_default)
    dt = _dt(p_default, prof)
    reg_d = generate_vortices(
        seed, bands, prof, p_default.storms, p_default.poles,
        dt=dt, dev_steps=p_default.sim.dev_steps,
    )
    reg_e = generate_vortices(
        seed, bands, prof, p_explicit.storms, p_explicit.poles,
        dt=dt, dev_steps=p_explicit.sim.dev_steps,
    )

    assert len(reg_d.vortices) == len(reg_e.vortices)
    for a, b in zip(reg_d.vortices, reg_e.vortices, strict=True):
        assert a.lat == b.lat
        assert a.lon == b.lon
        assert a.r_core == b.r_core
        assert a.strength == b.strength
        assert a.kind == b.kind


# ---------------------------------------------------------------- end-to-end drift


def test_hero_longitude_pins_the_rendered_longitude():
    """Pin, then actually run the per-step drift dev_steps times: the hero's
    final longitude equals the requested target (drift inverse-compensation)."""
    target_deg = 40.0
    p = PlanetParams()
    p.storms.hero_longitude = target_deg
    dev_steps = p.sim.dev_steps
    assert dev_steps > 0

    seed = 3
    bands, prof = _profiles(seed, p)
    dt = _dt(p, prof)
    reg = generate_vortices(
        seed, bands, prof, p.storms, poles=None, dt=dt, dev_steps=dev_steps
    )
    hero = _hero(reg)
    for _ in range(dev_steps):
        reg.drift(prof, dt)
    assert np.isclose(hero.lon, np.deg2rad(target_deg), atol=1e-3), (
        f"hero.lon={hero.lon!r}, expected {np.deg2rad(target_deg)!r}"
    )


def test_accent_longitude_pins_the_rendered_longitude():
    """First accent lands at the target after the dev run; a count=2 pair is a
    fixed 0.6 rad apart at the snapshot."""
    target_deg = -60.0
    p = PlanetParams()
    p.storms.accent_count = 2
    p.storms.accent_longitude = target_deg
    # Isolate the accents: no other populations, and mergers off so drift is the
    # only motion.
    p.storms.hero_count = 0
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.small_density = 0.0
    p.storms.merge_rate = 0.0
    dev_steps = p.sim.dev_steps

    seed = 11
    bands, prof = _profiles(seed, p)
    dt = _dt(p, prof)
    reg = generate_vortices(
        seed, bands, prof, p.storms, poles=None, dt=dt, dev_steps=dev_steps
    )
    accents = [v for v in reg.vortices if v.kind == KIND_OVAL]
    assert len(accents) == 2, f"expected 2 accents, got {len(accents)}"
    for _ in range(dev_steps):
        reg.drift(prof, dt)

    assert np.isclose(accents[0].lon, np.deg2rad(target_deg), atol=1e-3), (
        f"accent[0].lon={accents[0].lon!r}, expected {np.deg2rad(target_deg)!r}"
    )
    # Pair separation is the seeded fixed step (0.6 rad); both share a latitude
    # so drift keeps the gap constant.
    gap = (accents[1].lon - accents[0].lon + np.pi) % (2.0 * np.pi) - np.pi
    assert np.isclose(abs(gap), 0.6, atol=1e-3)


def test_outbreak_longitude_pins_the_head():
    """The pin overrides base_lon with a drift-compensated anchor: the head knot
    (k==0, largest radius) tracks that anchor within its ~0.02 rad seed jitter,
    and the anchor advected over its life lands exactly at the target. (The
    per-knot stagger + jitter are why the outbreak pin is best-effort, not exact
    like the hero/accent pins.)"""
    from gasgiant.sim.events import RADIUS, EventSchedule
    from gasgiant.sim.vortices import drift_compensated_lon, zonal_rate

    target_deg = 25.0
    lat_deg = 20.0
    p = PlanetParams()
    p.storms.outbreak_count = 1
    p.storms.outbreak_latitude = lat_deg  # pin lat so we know the drift rate
    p.storms.outbreak_longitude = target_deg
    p.storms.outbreak_phase = 0.6  # deterministic eruption step0
    dev_steps = p.sim.dev_steps

    seed = 5
    bands, prof = _profiles(seed, p)
    dt = _dt(p, prof)
    sched = EventSchedule.generate(p, bands, prof, dt)
    assert sched.outbreaks, "no outbreaks scheduled"

    # The head is the lead knot: k==0 gets the LEAD_RADIUS boost (largest).
    head = max(sched.outbreaks, key=lambda ob: ob.radius)
    assert head.radius > RADIUS  # the boosted lead knot

    # Reconstruct the compensated anchor the generator computed: step0 is
    # deterministic under a pinned phase; compensation is over (dev_steps-step0)
    # at the pinned latitude.
    step0 = int(0.6 * dev_steps)
    anchor = drift_compensated_lon(
        prof, float(np.deg2rad(lat_deg)), target_deg, dt, dev_steps - step0
    )
    # The head knot is base_lon plus only a ~N(0, 0.02) jitter (no k-stagger at
    # k==0), so it must track the anchor closely.
    dlon = (head.lon - anchor + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(dlon) < 0.1, f"head lon {head.lon!r} strayed from anchor {anchor!r}"

    # The anchor advected over its remaining life lands EXACTLY at the target —
    # this is the invariant the compensation guarantees.
    rate = float(zonal_rate(prof, np.array([np.deg2rad(lat_deg)]))[0])
    rendered = (anchor + rate * dt * (dev_steps - step0) + np.pi) % (2.0 * np.pi) - np.pi
    assert np.isclose(rendered, np.deg2rad(target_deg), atol=1e-6)


# ---------------------------------------------------------------- randomize


def test_randomize_skips_longitude_pins():
    """randomize must not touch the pins (no rand) and must not shift draws."""
    with_pins = PlanetParams()
    with_pins.storms.hero_longitude = 40.0
    with_pins.storms.accent_longitude = -60.0
    with_pins.storms.outbreak_longitude = 25.0

    r_plain = randomize(123, PlanetParams())
    r_with = randomize(123, with_pins)

    assert r_with.storms.hero_longitude == 40.0
    assert r_with.storms.accent_longitude == -60.0
    assert r_with.storms.outbreak_longitude == 25.0

    a = r_plain.model_dump()
    b = r_with.model_dump()
    b["storms"]["hero_longitude"] = None
    b["storms"]["accent_longitude"] = None
    b["storms"]["outbreak_longitude"] = None
    assert a == b


# ---------------------------------------------------------------- diff_tiers


def test_longitude_change_is_restart_tier():
    for field in ("hero_longitude", "accent_longitude", "outbreak_longitude"):
        a = PlanetParams()
        b = a.model_copy(deep=True)
        setattr(b.storms, field, 30.0)
        assert diff_tiers(a, b) == {Tier.RESTART}, field


# ---------------------------------------------------------------- panels


def test_panels_classify_optional_float():
    panels = pytest.importorskip("gasgiant.app.panels")
    info = StormsParams.model_fields["hero_longitude"]
    assert panels.leaf_kind("hero_longitude", info, None) == "optional_float"
    assert panels.leaf_kind("hero_longitude", info, 40.0) == "optional_float"


# ---------------------------------------------------------------- preset round-trip


def test_preset_roundtrip_with_longitude_pin(tmp_path):
    from gasgiant.params.presets import load_preset, save_preset

    p = PlanetParams()
    p.storms.hero_longitude = 40.0
    p.storms.accent_longitude = -60.0
    p.storms.outbreak_longitude = 25.0
    path = tmp_path / "lon.json"
    save_preset(p, path)
    loaded = load_preset(path)
    assert loaded.storms.hero_longitude == 40.0
    assert loaded.storms.accent_longitude == -60.0
    assert loaded.storms.outbreak_longitude == 25.0

    # None round-trips as None
    p2 = PlanetParams()
    save_preset(p2, path)
    reloaded = load_preset(path)
    assert reloaded.storms.hero_longitude is None
    assert reloaded.storms.accent_longitude is None
    assert reloaded.storms.outbreak_longitude is None

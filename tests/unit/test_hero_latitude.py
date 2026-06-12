"""hero_latitude: pin, validator, GUI classification, and params-walker tests."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, StormsParams, Tier
from gasgiant.params.randomize import randomize
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import KIND_HERO, generate_vortices

# ---------------------------------------------------------------- helpers

def _make_reg(seed: int, hero_latitude: float | None = None, dev_steps: int = 0):
    p = PlanetParams()
    if hero_latitude is not None:
        p.storms.hero_latitude = hero_latitude
    bands = generate_bands(p.seed if seed is None else seed, p.bands)
    profiles = build_profiles(p.seed if seed is None else seed, bands, p.bands, p.jets)
    return generate_vortices(
        p.seed if seed is None else seed,
        bands, profiles, p.storms, p.poles,
        dev_steps=dev_steps,
    )


def _make_reg_with_params(seed: int, storms: StormsParams):
    p = PlanetParams()
    bands = generate_bands(seed, p.bands)
    profiles = build_profiles(seed, bands, p.bands, p.jets)
    return generate_vortices(seed, bands, profiles, storms, p.poles)


def _hero(reg):
    heroes = [v for v in reg.vortices if v.kind == KIND_HERO]
    assert heroes, "no KIND_HERO vortex found"
    return heroes[0]


# ---------------------------------------------------------------- ST-0: None path unchanged

def test_default_none_leaves_fingerprint_path():
    """None hero_latitude must produce identical output to explicit None."""
    p_default = PlanetParams()
    p_explicit = PlanetParams()
    p_explicit.storms.hero_latitude = None

    assert p_default.storms.hero_latitude is None

    seed = 42
    bands_d = generate_bands(seed, p_default.bands)
    prof_d = build_profiles(seed, bands_d, p_default.bands, p_default.jets)
    reg_default = generate_vortices(seed, bands_d, prof_d, p_default.storms, p_default.poles)

    bands_e = generate_bands(seed, p_explicit.bands)
    prof_e = build_profiles(seed, bands_e, p_explicit.bands, p_explicit.jets)
    reg_explicit = generate_vortices(seed, bands_e, prof_e, p_explicit.storms, p_explicit.poles)

    assert len(reg_default.vortices) == len(reg_explicit.vortices)
    for a, b in zip(reg_default.vortices, reg_explicit.vortices, strict=True):
        assert np.isclose(a.lat, b.lat, atol=1e-10)
        assert np.isclose(a.lon, b.lon, atol=1e-10)
        assert np.isclose(a.r_core, b.r_core, atol=1e-10)
        assert np.isclose(a.strength, b.strength, atol=1e-10)


# ---------------------------------------------------------------- ST-1: pinning

def test_hero_pinned_to_exact_latitude():
    """All 5 seeds: hero lat == deg2rad(-22.5) exactly (atol 1e-6)."""
    target_deg = -22.5
    target_rad = float(np.deg2rad(target_deg))

    p = PlanetParams()
    p.storms.hero_latitude = target_deg

    for seed in range(5):
        bands = generate_bands(seed, p.bands)
        profiles = build_profiles(seed, bands, p.bands, p.jets)
        reg = generate_vortices(seed, bands, profiles, p.storms, p.poles, dev_steps=0)
        hero = _hero(reg)
        assert np.isclose(hero.lat, target_rad, atol=1e-6), (
            f"seed={seed}: hero.lat={hero.lat!r}, expected {target_rad!r}"
        )


# ---------------------------------------------------------------- validator

def test_validator_rejects_out_of_range():
    # cap for hero_radius=0.10: 63 - 206.3*0.10 = 42.37 -> 42.4
    with pytest.raises(ValidationError):
        StormsParams(hero_latitude=50.0, hero_radius=0.10)

    # -22.5 is within cap
    s = StormsParams(hero_latitude=-22.5, hero_radius=0.10)
    assert s.hero_latitude == -22.5

    # cap for hero_radius=0.25: 63 - 206.3*0.25 = 63 - 51.575 = 11.425
    s2 = StormsParams(hero_latitude=11.0, hero_radius=0.25)
    assert s2.hero_latitude == 11.0

    with pytest.raises(ValidationError):
        StormsParams(hero_latitude=12.0, hero_radius=0.25)


# ---------------------------------------------------------------- walker: panels

def test_panels_classify_optional_float_leaf():
    panels = pytest.importorskip("gasgiant.app.panels")
    info = StormsParams.model_fields["hero_latitude"]
    assert panels.leaf_kind("hero_latitude", info, None) == "optional_float"
    assert panels.leaf_kind("hero_latitude", info, -22.5) == "optional_float"


# ---------------------------------------------------------------- walker: randomize

def test_randomize_skips_hero_latitude():
    """randomize must not touch hero_latitude (no rand) and must not shift other draws."""
    with_lat = PlanetParams()
    with_lat.storms.hero_latitude = -22.5

    r_plain = randomize(123, PlanetParams())
    r_with = randomize(123, with_lat)

    # hero_latitude survives untouched
    assert r_with.storms.hero_latitude == -22.5

    # no draw consumed: all other fields identical
    a = r_plain.model_dump()
    b = r_with.model_dump()
    b["storms"]["hero_latitude"] = None
    assert a == b


# ---------------------------------------------------------------- walker: diff_tiers

def test_hero_latitude_change_is_restart_tier():
    a = PlanetParams()
    b = a.model_copy(deep=True)
    b.storms.hero_latitude = -22.5
    assert diff_tiers(a, b) == {Tier.RESTART}


# ---------------------------------------------------------------- walker: presets

def test_preset_roundtrip_with_hero_latitude(tmp_path):
    from gasgiant.params.presets import load_preset, save_preset

    p = PlanetParams()
    p.storms.hero_latitude = -22.5
    path = tmp_path / "hero.json"
    save_preset(p, path)
    loaded = load_preset(path)
    assert loaded.storms.hero_latitude == -22.5

    # None round-trips as None
    p2 = PlanetParams()
    save_preset(p2, path)
    assert load_preset(path).storms.hero_latitude is None

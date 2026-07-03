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


# ------------------------------------------------- B4-2: the pin-checkbox widget

def test_hero_latitude_cap_matches_validator():
    """The GUI slider bounds and the model validator share one formula."""
    from gasgiant.params.model import hero_latitude_cap

    assert hero_latitude_cap(0.10) == pytest.approx(63.0 - 206.3 * 0.10)
    # every legal hero_radius leaves a strictly positive symmetric cap, so
    # pinning at 0.0 is always valid
    assert hero_latitude_cap(0.25) > 0.0


def test_optional_float_bounds_clamp_to_radius_cap():
    panels = pytest.importorskip("gasgiant.app.panels")
    from gasgiant.params.model import hero_latitude_cap

    doc = {"hero_radius": 0.10, "hero_latitude": None}
    lo, hi = panels._optional_float_bounds("hero_latitude", doc, -55.0, 55.0)
    cap = hero_latitude_cap(0.10)
    assert (lo, hi) == (-cap, cap)

    # a small radius leaves the field bounds in charge
    doc["hero_radius"] = 0.03
    lo, hi = panels._optional_float_bounds("hero_latitude", doc, -55.0, 55.0)
    assert (lo, hi) == (-55.0, 55.0)

    # a non-hero optional float just uses the field bounds
    lo, hi = panels._optional_float_bounds("other", {}, -1.0, 2.0)
    assert (lo, hi) == (-1.0, 2.0)


@pytest.fixture
def imgui_ctx():
    imgui = pytest.importorskip("imgui_bundle.imgui")
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def test_pin_checkbox_pins_at_zero(imgui_ctx, monkeypatch):
    """Checking "pin" turns None into a committed 0.0 (always validator-legal)."""
    panels = pytest.importorskip("gasgiant.app.panels")
    imgui = imgui_ctx

    doc = {"hero_radius": 0.10, "hero_latitude": None}
    monkeypatch.setattr(panels.imgui, "checkbox", lambda label, v: (True, True))

    imgui.new_frame()
    imgui.begin("pin_test", None, 0)
    changed, committed = panels._draw_optional_float(
        "hero_latitude", "hero latitude", doc, -55.0, 55.0
    )
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (True, True)
    assert doc["hero_latitude"] == 0.0


def test_pin_checkbox_unpins_to_none(imgui_ctx, monkeypatch):
    panels = pytest.importorskip("gasgiant.app.panels")
    imgui = imgui_ctx

    doc = {"hero_radius": 0.10, "hero_latitude": -22.5}
    monkeypatch.setattr(panels.imgui, "checkbox", lambda label, v: (True, False))

    imgui.new_frame()
    imgui.begin("pin_test2", None, 0)
    changed, committed = panels._draw_optional_float(
        "hero_latitude", "hero latitude", doc, -55.0, 55.0
    )
    imgui.end()
    imgui.end_frame()

    assert (changed, committed) == (True, True)
    assert doc["hero_latitude"] is None


def test_pinned_slider_edits_value_within_cap(imgui_ctx, monkeypatch):
    """While pinned, the slider edits the value and its bounds are the
    radius-coupled cap, not the raw field bounds."""
    panels = pytest.importorskip("gasgiant.app.panels")
    from gasgiant.params.model import hero_latitude_cap
    imgui = imgui_ctx

    doc = {"hero_radius": 0.10, "hero_latitude": -22.5}
    seen_bounds = []

    monkeypatch.setattr(panels.imgui, "checkbox", lambda label, v: (False, v))

    def fake_slider(label, value, lo, hi, flags=0):
        seen_bounds.append((lo, hi))
        return True, -18.0

    monkeypatch.setattr(panels.imgui, "slider_float", fake_slider)

    imgui.new_frame()
    imgui.begin("pin_test3", None, 0)
    changed, committed = panels._draw_optional_float(
        "hero_latitude", "hero latitude", doc, -55.0, 55.0
    )
    imgui.end()
    imgui.end_frame()

    assert changed is True
    assert doc["hero_latitude"] == -18.0
    cap = hero_latitude_cap(0.10)
    assert seen_bounds == [(-cap, cap)]

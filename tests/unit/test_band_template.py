"""BandTemplate: validation, the verbatim generate_bands path, and the
params-system walkers (randomize / diff_tiers / panels / presets)."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import BandsParams, BandTemplate, PlanetParams, Tier
from gasgiant.params.randomize import randomize
from gasgiant.sim.bands import generate_bands

EDGES = [90.0, 45.0, 20.0, 5.0, -10.0, -35.0, -90.0]
VALUES = [0.70, 0.35, 0.75, 0.32, 0.68, 0.30]
HEIGHTS = [0.7, 0.4, 0.7, 0.4, 0.7, 0.4]


def _template(**over) -> BandTemplate:
    kw = {"edges_deg": EDGES, "values": VALUES, "heights": HEIGHTS}
    kw.update(over)
    return BandTemplate(**kw)


# ---------------------------------------------------------------- validation

def test_valid_template_accepted():
    t = _template()
    assert len(t.values) == len(t.edges_deg) - 1


@pytest.mark.parametrize(
    "over",
    [
        {"edges_deg": [90.0, 20.0, 45.0, -90.0], "values": [0.7, 0.3, 0.7],
         "heights": [0.7, 0.4, 0.7]},                      # not descending
        {"edges_deg": [80.0, 45.0, -90.0], "values": [0.7, 0.3],
         "heights": [0.7, 0.4]},                            # bad endpoint
        {"edges_deg": [90.0, 80.0, 0.0, -90.0], "values": [0.7, 0.3, 0.7],
         "heights": [0.7, 0.4, 0.7]},                       # interior > 76
        {"values": [0.70, 0.35, 0.75]},                     # length mismatch
        {"values": [0.70, 1.35, 0.75, 0.32, 0.68, 0.30]},   # out of range
        {"values": [0.70, 0.72, 0.30, 0.32, 0.68, 0.31]},   # adjacent zones
    ],
)
def test_invalid_templates_rejected(over):
    with pytest.raises(ValidationError):
        _template(**over)


def test_odd_belts_majority_impossible_is_caught():
    # 3 bands, belts in the majority: the median IS the top belt value, so
    # `values < median` cannot alternate -- the validator must say no.
    with pytest.raises(ValidationError):
        BandTemplate(
            edges_deg=[90.0, 30.0, -30.0, -90.0],
            values=[0.30, 0.75, 0.32],
            heights=[0.4, 0.7, 0.4],
        )


# ---------------------------------------------------------------- generation

def test_template_path_is_verbatim_and_seed_stable():
    params = BandsParams(template=_template())
    a = generate_bands(seed=1, params=params)
    b = generate_bands(seed=999, params=params)
    np.testing.assert_allclose(np.degrees(a.edges), EDGES, atol=1e-4)
    np.testing.assert_allclose(a.values, VALUES, atol=1e-6)
    np.testing.assert_allclose(a.heights, HEIGHTS, atol=1e-6)
    # No seasoning: layout identical across seeds...
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.heights, b.heights)
    # ...while the fade sector's own stream still varies by seed.
    assert a.fade_sector != b.fade_sector


def test_template_ignores_seasoning_knobs():
    base = BandsParams(template=_template())
    seasoned = BandsParams(
        template=_template(), value_contrast=1.8, hue_jitter=0.12,
        width_jitter=0.9, width_tail=0.7, count=20,
    )
    a = generate_bands(seed=7, params=base)
    b = generate_bands(seed=7, params=seasoned)
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.values, b.values)
    np.testing.assert_array_equal(a.heights, b.heights)


def test_template_fade_sector_targets_widest_belt():
    layout = generate_bands(seed=3, params=BandsParams(template=_template()))
    lat_lo, lat_hi, _, _ = layout.fade_sector
    # Belts (values < median 0.515) are 45..20, 5..-10, -35..-90; the last is
    # excluded by the |center| < 0.9 rad rule, so the widest eligible belt is
    # 45..20 (25 deg).
    assert np.isclose(np.degrees(lat_hi), 45.0, atol=1e-3)
    assert np.isclose(np.degrees(lat_lo), 20.0, atol=1e-3)


def test_default_none_is_the_seeded_path():
    a = generate_bands(seed=42, params=BandsParams())
    b = generate_bands(seed=42, params=BandsParams(template=None))
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.values, b.values)


# ---------------------------------------------------------------- walkers

def test_randomize_skips_template_and_shifts_no_draws():
    with_t = PlanetParams()
    with_t.bands.template = _template()
    r_plain = randomize(123, PlanetParams())
    r_with = randomize(123, with_t)
    # The template survives untouched...
    assert r_with.bands.template is not None
    assert r_with.bands.template.edges_deg == EDGES
    # ...and consumes no draw: every other randomized field is identical.
    a = r_plain.model_dump()
    b = r_with.model_dump()
    b["bands"]["template"] = None
    assert a == b


def test_template_change_is_restart_tier():
    a = PlanetParams()
    b = a.model_copy(deep=True)
    b.bands.template = _template()
    assert diff_tiers(a, b) == {Tier.RESTART}


def test_preset_roundtrip_with_template(tmp_path):
    from gasgiant.params.presets import load_preset, save_preset

    p = PlanetParams()
    p.bands.template = _template()
    path = tmp_path / "t.json"
    save_preset(p, path)
    loaded = load_preset(path)
    assert loaded.bands.template is not None
    assert loaded.bands.template.values == VALUES
    # And a templateless round-trip keeps None.
    p2 = PlanetParams()
    save_preset(p2, path)
    assert load_preset(path).bands.template is None


def test_panels_classify_the_template_leaf():
    panels = pytest.importorskip("gasgiant.app.panels")
    info = BandsParams.model_fields["template"]
    assert panels.leaf_kind("template", info, None) == "optional_model"
    assert panels.leaf_kind("template", info, _template().model_dump()) == "optional_model"

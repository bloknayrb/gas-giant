"""T14 per-band RGB tint: params + widget mapping (all GPU-free).

The forced-BAND_TINT no-op byte-identity and the latitude-row / tile behavior
live in tests/gpu/test_band_tint.py; here we cover the model defaults, the
strength-keyed no-op predicate, the stops-list widget mapping, and a preset
round-trip.
"""
from __future__ import annotations

import pytest

from gasgiant.params.model import AppearanceParams, GradientStop, PlanetParams


def _band_tint_on(appearance: AppearanceParams) -> bool:
    """Mirror of the render/maps.py predicate (strength > 0), which keys the
    BAND_TINT variant selection. Kept here so the silent-no-op trap is pinned
    without a GL context."""
    return appearance.band_tint_strength > 0.0


# -- defaults -------------------------------------------------------------------


def test_band_tint_defaults_are_no_op():
    a = AppearanceParams()
    assert a.band_tint_strength == 0.0
    # Non-empty NEUTRAL default: a flat mid-gray gradient.
    assert a.band_tint_stops
    for stop in a.band_tint_stops:
        assert stop.color == (0.5, 0.5, 0.5)


def test_appearance_stops_use_gradient_stop_verbatim():
    a = AppearanceParams()
    assert all(isinstance(s, GradientStop) for s in a.band_tint_stops)


def test_planet_params_composes_band_tint():
    a = PlanetParams().appearance
    assert isinstance(a, AppearanceParams)
    assert a.band_tint_strength == 0.0


# -- silent-no-op trap: predicate keys on strength ------------------------------


def test_strength_zero_is_band_tint_off():
    # Default (strength 0) selects the DEFAULT program => byte-identical.
    assert _band_tint_on(AppearanceParams()) is False


def test_nonneutral_stops_with_zero_strength_still_off():
    # The predicate must key on STRENGTH, not the stops: non-neutral stops with
    # strength 0 still compile the default program.
    a = AppearanceParams(
        band_tint_stops=[
            GradientStop(pos=0.0, color=(1.0, 0.0, 0.0)),
            GradientStop(pos=1.0, color=(0.0, 0.0, 1.0)),
        ]
    )
    assert _band_tint_on(a) is False


def test_positive_strength_is_band_tint_on():
    assert _band_tint_on(AppearanceParams(band_tint_strength=1e-6)) is True


# -- bounds + metadata ----------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_band_tint_strength_bounds(bad):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AppearanceParams(band_tint_strength=bad)


def test_band_tint_fields_carry_no_rand_metadata():
    # A rand draw on a POST art-direction override would reshuffle every later
    # randomize draw AND could silently recolor the planet.
    for name in ("band_tint_stops", "band_tint_strength"):
        extra = AppearanceParams.model_fields[name].json_schema_extra
        assert isinstance(extra, dict)
        assert "rand" not in extra, name


def test_band_tint_fields_have_literate_help():
    for name in ("band_tint_stops", "band_tint_strength"):
        assert AppearanceParams.model_fields[name].description


# -- widget mapping -------------------------------------------------------------


def test_leaf_kind_maps_band_tint_stops():
    panels = pytest.importorskip("gasgiant.app.panels")
    info = AppearanceParams.model_fields["band_tint_stops"]
    value = AppearanceParams().model_dump()["band_tint_stops"]
    assert panels.leaf_kind("band_tint_stops", info, value) == "stops"


# -- preset round-trip ----------------------------------------------------------


def test_band_tint_survives_model_round_trip():
    p = PlanetParams(name="tinted")
    p.appearance.band_tint_strength = 0.6
    p.appearance.band_tint_stops = [
        GradientStop(pos=0.0, color=(0.2, 0.3, 0.4)),
        GradientStop(pos=1.0, color=(0.9, 0.7, 0.5)),
    ]
    restored = PlanetParams.model_validate_json(p.model_dump_json())
    assert restored.appearance.band_tint_strength == 0.6
    assert restored.appearance.band_tint_stops[0].color == (0.2, 0.3, 0.4)
    assert restored.appearance.band_tint_stops[1].color == (0.9, 0.7, 0.5)

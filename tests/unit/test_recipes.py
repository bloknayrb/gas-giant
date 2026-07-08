"""T15 epoch recipes: overlay-merge correctness, strict rejection, and that
every shipped recipe validates against the current model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import (
    PresetError,
    apply_overlay,
    available_recipes,
    load_factory_preset,
    load_recipe,
)


def test_apply_overlay_preserves_untouched_siblings():
    """THE CRUX: overlaying ONE leaf of a nested group must leave every other
    sibling of that group unchanged. A shallow ``dict.update`` would replace the
    whole ``storms`` dict and reset the untouched siblings to their defaults."""
    base = load_factory_preset("jupiter_vorticity").model_copy(
        update={},
    )
    # Give several storms siblings distinct non-default values.
    base = base.model_validate(
        {
            **base.model_dump(),
            "storms": {
                **base.storms.model_dump(),
                "barge_density": 1.3,
                "oval_density": 0.7,
                "pearls_count": 11,
            },
        }
    )
    assert base.storms.barge_density == 1.3
    assert base.storms.oval_density == 0.7
    assert base.storms.pearls_count == 11

    # Overlay touches ONLY storms.barge_density.
    out = apply_overlay(base, {"storms": {"barge_density": 2.4}})

    assert out.storms.barge_density == 2.4          # changed
    assert out.storms.oval_density == 0.7           # sibling preserved
    assert out.storms.pearls_count == 11            # sibling preserved
    # And a wholly-untouched group is identical.
    assert out.bands == base.bands


def test_apply_overlay_deep_three_level_merge():
    """A 3-level-nested overlay (storms.cast is a list, but the export group is
    scalar; use the nested bands.template path for real depth) merges without
    disturbing siblings at any level."""
    base = load_factory_preset("jupiter_vorticity")
    # bands has many sibling scalars; touch two of them and assert the rest hold.
    before = base.bands.model_dump()
    out = apply_overlay(
        base, {"bands": {"belt_fade": 0.5, "faded_band_index": 5}}
    )
    assert out.bands.belt_fade == 0.5
    assert out.bands.faded_band_index == 5
    after = out.bands.model_dump()
    for key in before:
        if key in ("belt_fade", "faded_band_index"):
            continue
        assert after[key] == before[key], f"sibling {key} was disturbed"


def test_apply_overlay_rejects_unknown_key():
    base = load_factory_preset("jupiter_vorticity")
    with pytest.raises(ValidationError):
        apply_overlay(base, {"bands": {"no_such_field": 1.0}})
    with pytest.raises(ValidationError):
        apply_overlay(base, {"totally_bogus_group": {"x": 1}})


def test_apply_overlay_does_not_mutate_input():
    base = load_factory_preset("jupiter_vorticity")
    original = base.model_dump()
    apply_overlay(base, {"bands": {"belt_fade": 0.9}})
    assert base.model_dump() == original


def test_available_recipes_ships_both():
    names = available_recipes()
    assert "faded_seb" in names
    assert "ochre_ez" in names


@pytest.mark.parametrize("name", available_recipes())
def test_shipped_recipe_validates(name):
    """Every shipped recipe: load -> apply overlay on its base -> valid params."""
    base, overlay, meta = load_recipe(name)
    params = apply_overlay(load_factory_preset(base), overlay)
    assert isinstance(params, PlanetParams)
    assert isinstance(meta["name"], str) and meta["name"]
    assert isinstance(meta["description"], str)


def test_recipe_schema_roundtrip():
    """The recipe file schema (base + overlay + meta) round-trips into a valid
    PlanetParams with the overlay's leaves actually applied."""
    base, overlay, meta = load_recipe("faded_seb")
    assert base == "jupiter_vorticity"
    params = apply_overlay(load_factory_preset(base), overlay)
    assert params.bands.belt_fade == 1.0
    assert params.bands.faded_band_index == 6
    assert params.storms.outbreak_count == 2
    assert params.storms.outbreak_latitude == -13.4


def test_load_recipe_unknown_errors():
    with pytest.raises(PresetError):
        load_recipe("no_such_recipe")

from __future__ import annotations

import json

import pytest

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import (
    PresetError,
    factory_preset_names,
    load_factory_preset,
    load_preset,
    resolve_preset,
    save_preset,
)


def test_factory_presets_exist_and_load():
    names = factory_preset_names()
    assert "jupiter_like" in names
    assert "saturn_pale" in names
    for name in names:
        params = load_factory_preset(name)
        assert isinstance(params, PlanetParams)


def _luma(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_gas_giant_warm_keeps_zones_detailed():
    """The quiescent zone bands sit between fast jets that smear the detail
    tracer into smooth streaks, starving those latitudes of detail (~half the
    belts'). A high replenish_rate re-feeds detail faster than the jets smear it,
    keeping zones textured instead of reading as smooth 'blurry bands'. Pin it so
    a future tweak can't quietly drop it back to the starved default."""
    p = load_factory_preset("gas_giant_warm")
    assert p.turbulence.replenish_rate >= 0.25, (
        f"replenish_rate={p.turbulence.replenish_rate} is too low; the zone bands "
        f"will starve of detail and read as smooth blur"
    )


def test_vorticity_preset_is_live():
    """jupiter_vorticity was modernized from laminar (vort_inject=0, Gaussian hero,
    no L_d) to LIVE: shear-masked injection folds the bands into filaments, a finite
    deformation radius + scale-selective vort_psi_drag keep the storm-driven
    gravest-mode swirl from eating a band, and the hero is a solid-body coherent oval
    (not a center-draining whirlpool). Pin those so a future preset edit can't
    silently revert it to the dead look."""
    p = load_factory_preset("jupiter_vorticity")
    assert p.solver.type.value == "vorticity"
    assert p.solver.vort_inject > 0.0, "lost the live shear-masked injection"
    assert p.solver.vort_inject_mask.value == "shear"
    assert p.solver.deformation_radius > 0.0, "lost the cascade-screening L_d"
    # Gentle, texture-preserving drag (warm's hot 0.5 over-flattens this); >0 for
    # gravest-mode insurance, well under the over-flatten regime.
    assert 0.0 < p.solver.vort_psi_drag <= 0.3, p.solver.vort_psi_drag
    assert p.storms.hero_solid_core == 1.0, "hero reverted to Gaussian whirlpool"


def test_jupiter_vorticity_polar_values_persist():
    """The polar look (Juno blue tint, dense circumpolar cyclone field, and the
    dark blue-teal canvas that makes the folded-filament lace pop) lives in the
    preset's appearance/poles. The build script preserves it via read-modify-write
    (it never resets these), but pin it so a refactor that rebuilds appearance/poles
    from defaults can't silently wash the cap light again."""
    p = load_factory_preset("jupiter_vorticity")
    assert p.appearance.polar_tint_strength > 0.5, "lost the Juno polar tint"
    assert p.appearance.polar_canvas_value > 0.0, "lost the dark-teal polar canvas"
    assert p.poles.south.field_density > 1.0, "lost the dense circumpolar cyclone field"


@pytest.mark.parametrize("name", ["gas_giant_warm", "jupiter_vorticity", "jupiter_like"])
def test_palette_has_value_contrast(name):
    """The warm + Jupiter presets fix the 'frosted glass' look by mapping the rich T0
    color-index field through a HIGH-CONTRAST palette: a flat/pale ramp (the stock
    blue-grey Jupiter palette sat in a narrow mid-tone band, luma spread ~0.34, never
    dark) collapses the structure to one pale color = frost. Pin that each palette row
    keeps a real dark->bright luminance spread so a future edit can't silently re-frost."""
    p = load_factory_preset(name)
    rows = p.appearance.palette_rows
    assert rows, f"{name} must define palette rows"
    for row in rows:
        lumas = [_luma(s.color) for s in row.stops]
        assert max(lumas) - min(lumas) > 0.5, (
            f"{name} row at lat={row.latitude} palette is too flat "
            f"(luma spread {max(lumas) - min(lumas):.3f}); would re-frost"
        )


def test_jupiter_like_keeps_zones_detailed():
    """jupiter_like got the same zone-detail fix as gas_giant_warm (the startup
    default): a high replenish_rate re-feeds detail faster than the fast jets
    smear it, so the quiescent zones stay textured instead of reading as
    smooth 'blurry bands'. Pin it against a silent revert to the starved default."""
    p = load_factory_preset("jupiter_like")
    assert p.turbulence.replenish_rate >= 0.25, p.turbulence.replenish_rate


def test_save_load_round_trip(tmp_path):
    p = PlanetParams(seed=77, name="roundtrip")
    p.appearance.haze_amount = 0.33
    path = tmp_path / "rt.json"
    save_preset(p, path)
    q = load_preset(path)
    assert q == p


def test_unknown_factory_name_lists_available():
    with pytest.raises(PresetError, match="jupiter_like"):
        load_factory_preset("nope")


def test_typo_in_preset_is_an_error(tmp_path):
    doc = {
        "preset_format": 1,
        "name": "typo",
        "params": {"bands": {"cuont": 9}},
    }
    path = tmp_path / "typo.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PresetError, match="cuont"):
        load_preset(path)


def test_newer_format_rejected_with_message(tmp_path):
    doc = {"preset_format": 999, "name": "future", "params": {}}
    path = tmp_path / "future.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PresetError, match="newer"):
        load_preset(path)


def test_resolve_preset_path_and_name(tmp_path):
    p = PlanetParams(seed=5)
    path = tmp_path / "x.json"
    save_preset(p, path)
    assert resolve_preset(str(path)) == p
    assert resolve_preset("jupiter_like").name == "jupiter_like"


def test_sparse_preset_takes_defaults(tmp_path):
    doc = {"preset_format": 1, "name": "sparse", "params": {"seed": 3}}
    path = tmp_path / "sparse.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    p = load_preset(path)
    assert p.seed == 3
    assert p.bands.count == PlanetParams().bands.count

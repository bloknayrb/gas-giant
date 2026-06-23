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


def test_gas_giant_warm_palette_has_value_contrast():
    """gas_giant_warm fixes the 'frosted glass' look by mapping the rich T0
    color-index field through a HIGH-CONTRAST palette: a flat/pale palette
    discards the structure and the zones read as frosted. Pin that the bright
    and dark ends of each palette row keep a real luminance spread so a future
    palette edit can't silently re-flatten it back to frost."""
    p = load_factory_preset("gas_giant_warm")
    rows = p.appearance.palette_rows
    assert rows, "gas_giant_warm must define palette rows"

    def _luma(rgb: tuple[float, float, float]) -> float:
        r, g, b = rgb
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for row in rows:
        lumas = [_luma(s.color) for s in row.stops]
        # dark belt gaps -> bright cloud tops: a real value range, not a pale ramp.
        assert max(lumas) - min(lumas) > 0.5, (
            f"row at lat={row.latitude} palette is too flat "
            f"(luma spread {max(lumas) - min(lumas):.3f}); would re-frost"
        )


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

"""Preset format 1 -> 2: appearance.palette becomes appearance.palette_rows."""

from __future__ import annotations

import json

from gasgiant.params.migrations import CURRENT_PRESET_FORMAT
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_preset, load_preset_doc, to_preset_doc

_V1_PALETTE = [
    {"pos": 0.0, "color": [0.1, 0.2, 0.3]},
    {"pos": 1.0, "color": [0.9, 0.8, 0.7]},
]


def _v1_doc(params: dict) -> dict:
    return {"preset_format": 1, "app_version": "0.1.0", "name": "old", "params": params}


def test_current_format_is_2():
    assert CURRENT_PRESET_FORMAT == 2


def test_v1_palette_wraps_into_single_row():
    doc = _v1_doc({"appearance": {"palette": _V1_PALETTE, "haze_amount": 0.2}})
    p = load_preset_doc(doc)
    assert len(p.appearance.palette_rows) == 1
    row = p.appearance.palette_rows[0]
    assert row.latitude == 0.0
    assert [s.pos for s in row.stops] == [0.0, 1.0]
    assert row.stops[1].color == (0.9, 0.8, 0.7)
    assert p.appearance.haze_amount == 0.2  # sibling keys untouched


def test_v1_sparse_doc_without_appearance():
    p = load_preset_doc(_v1_doc({"seed": 7}))
    assert p.seed == 7
    assert p.appearance.palette_rows == PlanetParams().appearance.palette_rows


def test_v1_appearance_without_palette():
    p = load_preset_doc(_v1_doc({"appearance": {"haze_amount": 0.4}}))
    assert p.appearance.haze_amount == 0.4
    assert p.appearance.palette_rows == PlanetParams().appearance.palette_rows


def test_v1_session_shaped_doc(tmp_path):
    """The GUI autosave is an ordinary full-dump preset; a v1 one must load."""
    full = json.loads(PlanetParams(seed=11).to_json())
    full["appearance"]["palette"] = _V1_PALETTE
    del full["appearance"]["palette_rows"]
    path = tmp_path / "session.json"
    path.write_text(json.dumps(_v1_doc(full)), encoding="utf-8")
    p = load_preset(path)
    assert p.seed == 11
    assert p.appearance.palette_rows[0].stops[0].color == (0.1, 0.2, 0.3)


def test_saved_docs_carry_format_2():
    doc = to_preset_doc(PlanetParams(seed=1))
    assert doc["preset_format"] == 2
    assert "palette_rows" in doc["params"]["appearance"]
    assert "palette" not in doc["params"]["appearance"]

"""The vendored Blender reader must accept exactly what the exporter writes
(the CI guard for the exporter <-> add-on contract)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from gasgiant.export.manifest import build_manifest, write_manifest
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import to_preset_doc

ADDON_DIR = Path(__file__).resolve().parents[2] / "blender_addon" / "gasgiant_importer"


def _load_vendored_reader():
    spec = importlib.util.spec_from_file_location(
        "vendored_manifest_schema", ADDON_DIR / "manifest_schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_mapset(tmp_path: Path) -> Path:
    manifest = build_manifest(
        name="contract",
        seed=42,
        resolution=(2048, 1024),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "height": {"file": "height.exr", "format": "exr32f", "colorspace": "non-color"},
        },
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
        atmosphere_hint={"rim_color": [0.5, 0.6, 1.0], "rim_strength": 0.4},
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "height.exr").write_bytes(b"")
    return tmp_path


def test_vendored_reader_accepts_exporter_output(tmp_path):
    reader = _load_vendored_reader()
    doc = reader.read_mapset(_write_mapset(tmp_path))
    assert doc["name"] == "contract"
    assert doc["_warnings"] == []
    assert reader.map_path(doc, "color").name == "color.png"
    assert doc["physical"]["radius_km"] == 69911.0


def test_vendored_reader_resolves_emission_with_aurora_color(tmp_path):
    reader = _load_vendored_reader()
    manifest = build_manifest(
        name="contract",
        seed=42,
        resolution=(2048, 1024),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "emission": {
                "file": "emission.exr", "format": "exr32f",
                "colorspace": "non-color", "channels": 4,
                "aurora_color": [0.85, 0.35, 0.60],
            },
        },
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "emission.exr").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)
    assert reader.map_path(doc, "emission").name == "emission.exr"
    assert doc["maps"]["emission"]["aurora_color"] == [0.85, 0.35, 0.60]


def test_vendored_reader_emission_absent_is_none(tmp_path):
    reader = _load_vendored_reader()
    doc = reader.read_mapset(_write_mapset(tmp_path))
    assert reader.map_path(doc, "emission") is None


def test_vendored_reader_tolerates_unknown_keys(tmp_path):
    reader = _load_vendored_reader()
    path = _write_mapset(tmp_path) / "mapset.json"
    doc = json.loads(path.read_text())
    doc["future_field"] = {"anything": 1}
    doc["maps"]["color"]["future_attr"] = "x"
    path.write_text(json.dumps(doc))
    parsed = reader.read_mapset(tmp_path)
    assert parsed["name"] == "contract"


def test_vendored_reader_warns_on_newer_schema(tmp_path):
    reader = _load_vendored_reader()
    path = _write_mapset(tmp_path) / "mapset.json"
    doc = json.loads(path.read_text())
    doc["schema_version"] = 99
    path.write_text(json.dumps(doc))
    parsed = reader.read_mapset(tmp_path)
    assert any("newer" in w for w in parsed["_warnings"])


def test_vendored_reader_rejects_missing_color(tmp_path):
    reader = _load_vendored_reader()
    path = _write_mapset(tmp_path) / "mapset.json"
    doc = json.loads(path.read_text())
    del doc["maps"]["color"]
    path.write_text(json.dumps(doc))
    with pytest.raises(reader.MapsetError):
        reader.read_mapset(tmp_path)

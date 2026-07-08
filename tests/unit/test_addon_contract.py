"""The vendored Blender reader must accept exactly what the exporter writes
(the CI guard for the exporter <-> add-on contract)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from gasgiant.export.manifest import attach_frames, build_manifest, write_manifest
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


def test_vendored_reader_resolves_flow_with_convention(tmp_path):
    """T10: the additive `flow` map (channels 4 + a `convention` string) resolves
    through the tolerant vendored reader with no warnings; the convention travels
    untouched so the importer can interpret the RG = (east, north) layout."""
    reader = _load_vendored_reader()
    manifest = build_manifest(
        name="contract",
        seed=42,
        resolution=(2048, 1024),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "flow": {
                "file": "flow.exr", "format": "exr32f",
                "colorspace": "non-color", "channels": 4,
                "convention": "rg_east_north_texel_per_step",
            },
        },
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "flow.exr").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)
    assert doc["_warnings"] == []
    assert reader.map_path(doc, "flow").name == "flow.exr"
    assert doc["maps"]["flow"]["convention"] == "rg_east_north_texel_per_step"


def test_vendored_reader_resolves_rings_with_extent(tmp_path):
    """T16: the additive `rings` map (channels 4 + a `convention` string) plus the
    physical.ring_*_km fields resolve through the tolerant vendored reader with no
    warnings; ring_extent() returns the exported span for the annulus builder."""
    reader = _load_vendored_reader()
    manifest = build_manifest(
        name="contract",
        seed=42,
        resolution=(2048, 1024),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "rings": {
                "file": "rings.exr", "format": "exr32f",
                "colorspace": "non-color", "channels": 4,
                "convention": "radial_inner_to_outer_alpha_coverage",
            },
        },
        physical={
            "radius_km": 60268.0, "height_scale": 0.004, "height_midlevel": 0.5,
            "ring_inner_km": 74500.0, "ring_outer_km": 136780.0,
        },
        preset_doc=to_preset_doc(PlanetParams()),
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "rings.exr").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)
    assert doc["_warnings"] == []
    assert reader.map_path(doc, "rings").name == "rings.exr"
    assert reader.ring_extent(doc) == (74500.0, 136780.0)


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


def test_vendored_reader_accepts_all_maps_frames_block(tmp_path):
    """T7: the additive frames.maps / frames.video sub-blocks (all-maps + mp4
    sequence export) don't break the vendored tolerant reader — the base map set
    still resolves cleanly with no warnings."""
    reader = _load_vendored_reader()
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
    )
    attach_frames(
        manifest, count=3, steps_per_frame=8,
        files=[f"frames/frame_{i:04d}.png" for i in range(3)],
        maps={
            "height": [f"frames/height_{i:04d}.png" for i in range(3)],
            "emission": [f"frames/emission_{i:04d}.exr" for i in range(3)],
        },
        video="sequence.mp4",
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "height.exr").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)
    assert doc["_warnings"] == []
    assert reader.map_path(doc, "color").name == "color.png"
    # the frames block travels through untouched (readers may use it or ignore it)
    assert doc["frames"]["maps"]["emission"][2] == "frames/emission_0002.exr"
    assert doc["frames"]["video"] == "sequence.mp4"


def test_vendored_reader_frame_helpers_resolve_sequence_paths(tmp_path):
    """T9: the reader's frame-path helpers pick the frame-0 file of each animated
    map from real exporter output (color from frames.files, height/emission from
    frames.maps), so the importer can load the sequence head."""
    reader = _load_vendored_reader()
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
    )
    attach_frames(
        manifest, count=5, steps_per_frame=8,
        files=[f"frames/frame_{i:04d}.png" for i in range(5)],
        maps={
            "height": [f"frames/height_{i:04d}.png" for i in range(5)],
            "emission": [f"frames/emission_{i:04d}.exr" for i in range(5)],
        },
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "height.exr").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)

    assert reader.frame_count(doc) == 5
    assert reader.frame_zero_path(doc, "color").name == "frame_0000.png"
    assert reader.frame_zero_path(doc, "height").name == "height_0000.png"
    assert reader.frame_zero_path(doc, "emission").name == "emission_0000.exr"


def test_vendored_reader_frame_helpers_none_without_frames(tmp_path):
    """A still map set (no frames block) reports no sequence."""
    reader = _load_vendored_reader()
    doc = reader.read_mapset(_write_mapset(tmp_path))
    assert reader.frames_block(doc) is None
    assert reader.frame_count(doc) == 0
    assert reader.frame_zero_path(doc, "color") is None

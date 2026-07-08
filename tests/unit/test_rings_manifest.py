"""T16: the additive `rings` map entry + physical.ring_*_km round-trip through
the vendored (stdlib-only) Blender reader, and the manifest still validates
against the schema (no schema_version bump -- rings are additive)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from gasgiant.export.manifest import build_manifest, write_manifest
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import to_preset_doc

ADDON_DIR = Path(__file__).resolve().parents[2] / "blender_addon" / "gasgiant_importer"


def _load_vendored_reader():
    spec = importlib.util.spec_from_file_location(
        "vendored_manifest_schema_rings", ADDON_DIR / "manifest_schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rings_manifest():
    return build_manifest(
        name="ringed",
        seed=3,
        resolution=(2048, 1024),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "height": {"file": "height.exr", "format": "exr32f", "colorspace": "non-color"},
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


def test_rings_manifest_validates_against_schema():
    """build_manifest self-validates: an additive rings map + ring_*_km physical
    fields pass the schema with schema_version still 1."""
    m = _rings_manifest()
    assert m["schema_version"] == 1
    assert m["maps"]["rings"]["channels"] == 4
    assert m["physical"]["ring_inner_km"] == 74500.0
    assert m["physical"]["ring_outer_km"] == 136780.0


def test_vendored_reader_resolves_rings_and_extent(tmp_path):
    reader = _load_vendored_reader()
    write_manifest(tmp_path, _rings_manifest())
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "height.exr").write_bytes(b"")
    (tmp_path / "rings.exr").write_bytes(b"")

    doc = reader.read_mapset(tmp_path)
    assert doc["_warnings"] == []
    assert reader.map_path(doc, "rings").name == "rings.exr"
    assert doc["maps"]["rings"]["convention"] == "radial_inner_to_outer_alpha_coverage"
    assert reader.ring_extent(doc) == (74500.0, 136780.0)


def test_ring_extent_none_without_rings_map(tmp_path):
    reader = _load_vendored_reader()
    manifest = build_manifest(
        name="plain",
        seed=1,
        resolution=(2048, 1024),
        maps={"color": {"file": "color.png", "format": "png16", "colorspace": "srgb"}},
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "color.png").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)
    assert reader.ring_extent(doc) is None

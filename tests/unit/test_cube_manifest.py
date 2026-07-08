"""T17: the cube-map manifest (schema v2) and the exporter<->addon contract.

- v1 equirect manifest is UNCHANGED (schema_version 1, projection
  "equirectangular", per-map ``file``) so deployed importers never warn.
- v2 cube manifest carries schema_version 2, projection "cube", and a per-map
  ``faces`` block (file XOR faces, enforced by the schema).
- the vendored (stdlib-only) Blender reader accepts v1 and REJECTS a cube set
  with a clear message (this importer builds no cube geometry).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jsonschema
import pytest

from gasgiant.export.manifest import (
    CUBE_FACE_NAMES,
    PROJECTION_CUBE,
    PROJECTION_EQUIRECT,
    build_manifest,
    load_schema,
    read_manifest,
    write_manifest,
)
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import to_preset_doc

ADDON_DIR = Path(__file__).resolve().parents[2] / "blender_addon" / "gasgiant_importer"


def _load_vendored_reader():
    spec = importlib.util.spec_from_file_location(
        "vendored_manifest_schema_cube", ADDON_DIR / "manifest_schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _faces(prefix: str, ext: str) -> dict[str, str]:
    return {fn: f"{prefix}_{fn}.{ext}" for fn in CUBE_FACE_NAMES}


def _equirect_manifest() -> dict:
    return build_manifest(
        name="eq",
        seed=1,
        resolution=(2048, 1024),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "height": {"file": "height.exr", "format": "exr32f", "colorspace": "non-color"},
        },
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
    )


def _cube_manifest() -> dict:
    return build_manifest(
        name="cube",
        seed=1,
        resolution=(512, 512),
        maps={
            "color": {
                "faces": _faces("color", "png"), "format": "png16",
                "colorspace": "srgb", "channels": 3,
            },
            "height": {
                "faces": _faces("height", "exr"), "format": "exr32f",
                "colorspace": "non-color", "channels": 1,
            },
        },
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
        projection=PROJECTION_CUBE,
    )


def test_equirect_manifest_unchanged():
    """The default (equirect) manifest must stay schema_version 1 with a plain
    ``file`` per map and the legacy 'equirectangular' projection string."""
    m = _equirect_manifest()
    assert m["schema_version"] == 1
    assert m["projection"] == PROJECTION_EQUIRECT == "equirectangular"
    assert m["maps"]["color"]["file"] == "color.png"
    assert "faces" not in m["maps"]["color"]


def test_cube_manifest_is_v2_with_faces():
    m = _cube_manifest()
    assert m["schema_version"] == 2
    assert m["projection"] == "cube"
    for name in ("color", "height"):
        entry = m["maps"][name]
        assert "file" not in entry
        assert set(entry["faces"]) == set(CUBE_FACE_NAMES)
    assert m["maps"]["color"]["faces"]["px"] == "color_px.png"


def test_cube_manifest_faces_round_trip(tmp_path):
    """Writer -> reader round-trip: the faces block survives write_manifest +
    read_manifest (which re-validates against the schema)."""
    m = _cube_manifest()
    write_manifest(tmp_path, m)
    back = read_manifest(tmp_path)
    assert back["projection"] == "cube"
    assert back["schema_version"] == 2
    assert back["maps"]["height"]["faces"]["nz"] == "height_nz.exr"
    assert back["maps"]["color"]["faces"] == _faces("color", "png")


def test_schema_rejects_file_and_faces_together():
    """A maps entry must be file XOR faces -- both present is invalid."""
    m = _cube_manifest()
    m["maps"]["color"]["file"] = "color.png"  # now has BOTH file and faces
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, load_schema())


def test_schema_rejects_entry_with_neither_file_nor_faces():
    m = _equirect_manifest()
    del m["maps"]["color"]["file"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, load_schema())


def test_vendored_reader_accepts_equirect(tmp_path):
    reader = _load_vendored_reader()
    write_manifest(tmp_path, _equirect_manifest())
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "height.exr").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)
    assert doc["_warnings"] == []
    assert reader.map_path(doc, "color").name == "color.png"


def test_vendored_reader_rejects_cube_clearly(tmp_path):
    """This importer builds no cube geometry: a cube mapset must be rejected with
    a clear, actionable message that names the projection and points at a newer
    importer."""
    reader = _load_vendored_reader()
    write_manifest(tmp_path, _cube_manifest())
    for fn in CUBE_FACE_NAMES:
        (tmp_path / f"color_{fn}.png").write_bytes(b"")
        (tmp_path / f"height_{fn}.exr").write_bytes(b"")
    with pytest.raises(reader.MapsetError) as exc:
        reader.read_mapset(tmp_path)
    msg = str(exc.value).lower()
    assert "cube" in msg
    assert "importer" in msg

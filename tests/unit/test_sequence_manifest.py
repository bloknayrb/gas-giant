"""The additive frames.maps / frames.video sub-blocks (T7 all-maps / video
sequence export) validate against the schema and stay readable by the vendored,
tolerant Blender reader."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from gasgiant.export.manifest import attach_frames, build_manifest, read_manifest, write_manifest
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import to_preset_doc

ADDON_DIR = Path(__file__).resolve().parents[2] / "blender_addon" / "gasgiant_importer"


def _load_vendored_reader():
    spec = importlib.util.spec_from_file_location(
        "vendored_manifest_schema_seq", ADDON_DIR / "manifest_schema.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest():
    return build_manifest(
        name="seq",
        seed=1,
        resolution=(512, 256),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "height": {"file": "height.exr", "format": "exr32f", "colorspace": "non-color"},
        },
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
    )


def test_attach_frames_all_maps_writes_maps_block(tmp_path):
    m = _manifest()
    attach_frames(
        m, count=3, steps_per_frame=8,
        files=[f"frames/frame_{i:04d}.png" for i in range(3)],
        maps={
            "height": [f"frames/height_{i:04d}.png" for i in range(3)],
            "emission": [f"frames/emission_{i:04d}.exr" for i in range(3)],
        },
    )
    write_manifest(tmp_path, m)
    back = read_manifest(tmp_path)  # re-validates against the schema
    assert back["frames"]["maps"]["height"] == [f"frames/height_{i:04d}.png" for i in range(3)]
    assert back["frames"]["maps"]["emission"] == [f"frames/emission_{i:04d}.exr" for i in range(3)]
    # color list untouched by the additive block
    assert back["frames"]["files"] == [f"frames/frame_{i:04d}.png" for i in range(3)]


def test_attach_frames_video_key(tmp_path):
    m = _manifest()
    attach_frames(
        m, count=2, steps_per_frame=4,
        files=[f"frames/frame_{i:04d}.png" for i in range(2)],
        video="sequence.mp4",
    )
    write_manifest(tmp_path, m)
    assert read_manifest(tmp_path)["frames"]["video"] == "sequence.mp4"


def test_attach_frames_without_maps_omits_block():
    m = _manifest()
    attach_frames(m, count=2, steps_per_frame=4, files=["frames/frame_0000.png",
                                                         "frames/frame_0001.png"])
    assert "maps" not in m["frames"]
    assert "video" not in m["frames"]


def test_vendored_reader_tolerates_maps_block(tmp_path):
    """The stdlib Blender reader ignores the new frames.maps / frames.video keys
    (forward compatibility) and still resolves the base maps."""
    reader = _load_vendored_reader()
    m = _manifest()
    attach_frames(
        m, count=2, steps_per_frame=4,
        files=[f"frames/frame_{i:04d}.png" for i in range(2)],
        maps={"height": [f"frames/height_{i:04d}.png" for i in range(2)]},
        video="sequence.mp4",
    )
    write_manifest(tmp_path, m)
    (tmp_path / "color.png").write_bytes(b"")
    (tmp_path / "height.exr").write_bytes(b"")
    doc = reader.read_mapset(tmp_path)
    assert doc["name"] == "seq"
    assert doc["_warnings"] == []
    assert doc["frames"]["maps"]["height"][0] == "frames/height_0000.png"


def test_attach_frames_bad_maps_list_rejected():
    """A maps sub-list with an empty file list fails schema validation."""
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        attach_frames(
            _manifest(), count=2, steps_per_frame=4,
            files=["frames/frame_0000.png", "frames/frame_0001.png"],
            maps={"height": []},
        )

from __future__ import annotations

import pytest

from gasgiant.export.manifest import build_manifest, load_schema, read_manifest, write_manifest
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import to_preset_doc


def _manifest():
    return build_manifest(
        name="test",
        seed=1,
        resolution=(2048, 1024),
        maps={
            "color": {"file": "color.png", "format": "png16", "colorspace": "srgb"},
            "height": {"file": "height.exr", "format": "exr32f", "colorspace": "non-color"},
        },
        physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
        preset_doc=to_preset_doc(PlanetParams()),
    )


def test_build_validates_against_schema():
    m = _manifest()
    assert m["schema_version"] == 1
    assert m["projection"] == "equirectangular"


def test_emission_entry_validates_against_schema():
    m = build_manifest(
        name="test",
        seed=1,
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
    assert m["maps"]["emission"]["aurora_color"] == [0.85, 0.35, 0.60]


def test_invalid_manifest_rejected():
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        build_manifest(
            name="bad",
            seed=1,
            resolution=(2048, 1024),
            maps={"height": {"file": "h.exr", "format": "exr32f", "colorspace": "non-color"}},
            physical={"radius_km": 69911.0, "height_scale": 0.004, "height_midlevel": 0.5},
            preset_doc={},
        )  # missing required "color" map


def test_write_read_round_trip(tmp_path):
    m = _manifest()
    write_manifest(tmp_path, m)
    back = read_manifest(tmp_path)
    assert back == m


def test_schema_loads():
    schema = load_schema()
    assert schema["title"].startswith("Gas Giant")


def test_frames_block_attach_and_round_trip(tmp_path):
    from gasgiant.export.manifest import attach_frames

    m = _manifest()
    attach_frames(
        m, count=3, steps_per_frame=8,
        files=[f"frames/frame_{i:04d}.png" for i in range(3)],
    )
    write_manifest(tmp_path, m)
    back = read_manifest(tmp_path)  # read side re-validates against the schema
    assert back["frames"]["count"] == 3
    assert back["frames"]["steps_per_frame"] == 8
    assert back["frames"]["files"] == [f"frames/frame_{i:04d}.png" for i in range(3)]


def test_frames_block_invalid_rejected():
    import jsonschema

    from gasgiant.export.manifest import attach_frames

    with pytest.raises(jsonschema.ValidationError):
        attach_frames(_manifest(), count=0, steps_per_frame=8, files=[])


def test_default_manifest_has_no_frames_block():
    assert "frames" not in _manifest()


def test_embedded_preset_round_trips():
    m = _manifest()
    from gasgiant.params.presets import load_preset_doc

    params = load_preset_doc(m["preset"])
    assert params == PlanetParams()

"""The stdlib sequence fixture-writer (shared by the Blender background test)
must produce a map set the vendored reader accepts, with the `frames` block and
the new frame-path helpers round-tripping. GPU-free and bpy-free — this is the
part of T9 that runs outside Blender."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADDON_DIR = ROOT / "blender_addon" / "gasgiant_importer"
BLENDER_TEST = ROOT / "tests" / "blender" / "test_import.py"


def _load_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reader():
    return _load_from_path("vendored_manifest_schema", ADDON_DIR / "manifest_schema.py")


def _fixture_module():
    # test_import.py guards its bpy run behind `if __name__ == "__main__"`, so
    # importing it here does NOT touch bpy — only the stdlib fixture writer.
    return _load_from_path("gg_blender_test_import", BLENDER_TEST)


def test_fixture_writer_needs_no_bpy():
    mod = _fixture_module()
    assert callable(mod.write_sequence_fixture)
    assert "bpy" not in sys.modules  # importing the module must not pull bpy in


def test_fixture_is_accepted_by_vendored_reader(tmp_path):
    reader = _reader()
    mod = _fixture_module()
    mod.write_sequence_fixture(tmp_path, count=3)

    doc = reader.read_mapset(tmp_path)
    assert doc["name"] == "seqfix"
    assert doc["_warnings"] == []  # every referenced still map exists
    assert reader.map_path(doc, "color").name == "color.png"


def test_frames_block_round_trips(tmp_path):
    reader = _reader()
    mod = _fixture_module()
    mod.write_sequence_fixture(tmp_path, count=3, steps_per_frame=8)

    doc = reader.read_mapset(tmp_path)
    frames = reader.frames_block(doc)
    assert frames is not None
    assert frames["count"] == 3
    assert frames["steps_per_frame"] == 8
    assert frames["files"][0] == "frames/frame_0000.png"
    assert frames["maps"]["emission"][2] == "frames/emission_0002.exr"
    assert frames["video"] == "sequence.mp4"
    assert reader.frame_count(doc) == 3


def test_frame_zero_paths(tmp_path):
    reader = _reader()
    mod = _fixture_module()
    mod.write_sequence_fixture(tmp_path, count=3)

    doc = reader.read_mapset(tmp_path)
    assert reader.frame_zero_path(doc, "color").name == "frame_0000.png"
    assert reader.frame_zero_path(doc, "height").name == "height_0000.png"
    assert reader.frame_zero_path(doc, "emission").name == "emission_0000.exr"
    # frame-0 files really exist on disk (Blender loads them as the sequence head)
    assert reader.frame_zero_path(doc, "color").is_file()
    assert reader.frame_zero_path(doc, "height").is_file()


def test_still_mapset_has_no_frames(tmp_path):
    reader = _reader()
    # a manifest with no frames block -> helpers report "still"
    mod = _fixture_module()
    mod.write_sequence_fixture(tmp_path, count=2)
    doc = reader.read_mapset(tmp_path)
    del doc["frames"]
    assert reader.frames_block(doc) is None
    assert reader.frame_count(doc) == 0
    assert reader.frame_zero_path(doc, "color") is None

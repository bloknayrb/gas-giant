"""mapset.json: the exporter <-> Blender importer contract.

The canonical JSON Schema lives next to this module (mapset.schema.json);
the Blender add-on vendors a stdlib-only reader that CI validates against the
same schema. Reader policy is TOLERANT (unknown keys ignored) — the opposite
of presets, see docs/presets.md.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

import gasgiant

SCHEMA_VERSION = 1
# Cube-map export (T17) is a distinct schema tier: each maps entry carries a
# per-face ``faces`` block instead of a single ``file``, and the top-level
# ``projection`` becomes "cube". Readers gate on ``projection`` (a cube-unaware
# importer rejects it cleanly), so the version bump is informational.
SCHEMA_VERSION_CUBE = 2
MANIFEST_FILENAME = "mapset.json"

# Manifest ``projection`` string per ExportParams.projection kind. Equirect keeps
# the legacy full word "equirectangular" (schema v1, UNCHANGED so deployed
# add-ons never warn); cube is the new v2 value.
PROJECTION_EQUIRECT = "equirectangular"
PROJECTION_CUBE = "cube"

# Cube-map face names, in the standard GL cube-map axis order (+X,-X,+Y,-Y,+Z,-Z).
# The manifest ``faces`` block and the exported ``<map>_<face>.<ext>`` filenames
# both use these; the validator's edge-continuity table indexes this same order.
CUBE_FACE_NAMES = ("px", "nx", "py", "ny", "pz", "nz")


def load_schema() -> dict[str, Any]:
    ref = resources.files("gasgiant.export") / "mapset.schema.json"
    return json.loads(ref.read_text(encoding="utf-8"))


def build_manifest(
    *,
    name: str,
    seed: int,
    resolution: tuple[int, int],
    maps: dict[str, dict[str, Any]],
    physical: dict[str, float],
    preset_doc: dict[str, Any],
    atmosphere_hint: dict[str, Any] | None = None,
    projection: str = PROJECTION_EQUIRECT,
) -> dict[str, Any]:
    """``projection`` selects the output-map layout. The default
    ("equirectangular") is the legacy path: ``schema_version`` stays 1 and each
    ``maps`` entry carries a single ``file`` -- byte-for-byte the pre-T17 manifest,
    so deployed importers never warn. "cube" bumps ``schema_version`` to 2 and
    each ``maps`` entry instead carries a ``faces`` block (the 6 per-face files);
    the schema enforces file-XOR-faces per entry."""
    is_cube = projection == PROJECTION_CUBE
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_CUBE if is_cube else SCHEMA_VERSION,
        "generator": {"name": "gasgiant", "version": gasgiant.__version__},
        "name": name,
        "seed": seed,
        "projection": projection,
        "resolution": list(resolution),
        "physical": physical,
        "maps": maps,
        "preset": preset_doc,
    }
    if atmosphere_hint:
        manifest["atmosphere_hint"] = atmosphere_hint
    jsonschema.validate(manifest, load_schema())
    return manifest


def attach_frames(
    manifest: dict[str, Any],
    *,
    count: int,
    steps_per_frame: int,
    files: list[str],
    maps: dict[str, list[str]] | None = None,
    video: str | None = None,
) -> dict[str, Any]:
    """Attach the optional animation ``frames`` block to a built manifest and
    re-validate (the writer self-validates; readers stay tolerant).

    ``files`` is the colour-frame file list (always present). ``maps`` is an
    optional per-map file-list sub-block written by an ``all_maps`` sequence
    export (e.g. ``{"height": [...], "emission": [...]}``); ``video`` is the
    relative path of an encoded mp4. Both are ADDITIVE -- older tolerant readers
    ignore ``frames.maps`` / ``frames.video`` and consume ``frames.files`` as
    before, so no ``schema_version`` bump."""
    block: dict[str, Any] = {
        "count": count,
        "steps_per_frame": steps_per_frame,
        "pattern": "frames/frame_%04d.png",
        "files": files,
    }
    if maps:
        block["maps"] = maps
    if video:
        block["video"] = video
    manifest["frames"] = block
    jsonschema.validate(manifest, load_schema())
    return manifest


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_manifest(mapset_dir: Path) -> dict[str, Any]:
    path = mapset_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {MANIFEST_FILENAME} in {mapset_dir}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(manifest, load_schema())
    return manifest



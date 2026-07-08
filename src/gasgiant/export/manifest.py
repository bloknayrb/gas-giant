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
MANIFEST_FILENAME = "mapset.json"


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
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "gasgiant", "version": gasgiant.__version__},
        "name": name,
        "seed": seed,
        "projection": "equirectangular",
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



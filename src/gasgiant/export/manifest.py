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
from gasgiant.params.presets import to_preset_doc

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


def export_mapset(sim: Any, out_dir: Path) -> Path:
    """Render and write a complete map set + manifest. Phase 1: one-shot,
    full-resolution render (tiled snapshot export replaces this in Phase 4).

    ``sim`` is an engine.Simulation; typed as Any to keep export below engine
    in the layer order (engine calls this with itself).
    """
    from gasgiant.export.writers import write_exr_gray, write_png16_rgb

    p = sim.params
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = sim.render_maps(p.export.width)

    write_png16_rgb(out_dir / "color.png", rendered["color"], p.export.png_compression)
    write_exr_gray(out_dir / "height.exr", rendered["height"])

    maps = {
        "color": {"file": "color.png", "format": "png16", "colorspace": "srgb", "channels": 3},
        "height": {
            "file": "height.exr", "format": "exr32f", "colorspace": "non-color", "channels": 1,
        },
    }
    manifest = build_manifest(
        name=p.name,
        seed=p.seed,
        resolution=(p.export.width, p.export.width // 2),
        maps=maps,
        physical={
            "radius_km": p.physical.radius_km,
            "height_scale": p.physical.height_scale,
            "height_midlevel": p.physical.height_midlevel,
        },
        preset_doc=to_preset_doc(p),
        atmosphere_hint={"rim_color": [0.55, 0.65, 1.0], "rim_strength": 0.4},
    )
    return write_manifest(out_dir, manifest)

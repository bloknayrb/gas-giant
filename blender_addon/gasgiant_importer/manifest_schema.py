"""Vendored, stdlib-only mapset.json reader.

The authoritative schema lives in the gasgiant package
(src/gasgiant/export/mapset.schema.json); CI checks this reader accepts what
the exporter writes. Policy here is TOLERANT: unknown keys are ignored
(forward compatibility), missing optional maps are skipped with a warning —
the opposite of the strict preset policy, by design.
"""

from __future__ import annotations

import json
from pathlib import Path

SUPPORTED_SCHEMA = 1

REQUIRED_KEYS = ("schema_version", "name", "seed", "projection", "resolution", "maps")


class MapsetError(ValueError):
    pass


def read_mapset(path: Path) -> dict:
    """Load and minimally validate a mapset.json (or its containing folder).

    Returns the manifest dict with an added "_dir" key (absolute folder) and
    "_warnings" list.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "mapset.json"
    if not path.is_file():
        raise MapsetError(f"no mapset.json at {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MapsetError(f"{path.name}: not valid JSON ({exc})") from exc

    warnings: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in doc:
            raise MapsetError(f"{path.name}: missing required key {key!r}")

    version = doc.get("schema_version", 1)

    # Projection gate. This importer builds ONLY equirectangular geometry, so a
    # cube-map set (schema v2, projection "cube", per-map "faces" blocks) is
    # rejected with a clear, actionable message rather than a confusing
    # missing-'file' failure deeper in. Cube support needs a newer importer.
    projection = doc.get("projection")
    if projection == "cube":
        raise MapsetError(
            f"cube-map mapsets (schema_version {version}, projection 'cube') require "
            f"a newer Gas Giant importer that builds cube geometry; this add-on "
            f"supports the equirectangular projection only (schema <= {SUPPORTED_SCHEMA}). "
            f"Re-export with projection=equirect, or update the importer."
        )
    if projection != "equirectangular":
        raise MapsetError(f"unsupported projection {projection!r}")

    if version > SUPPORTED_SCHEMA:
        warnings.append(
            f"mapset schema_version {version} is newer than this add-on understands "
            f"({SUPPORTED_SCHEMA}); importing best-effort"
        )

    maps = doc.get("maps", {})
    if "color" not in maps:
        raise MapsetError("mapset has no color map")

    folder = path.parent.resolve()
    for name in list(maps):
        entry = maps[name]
        file = folder / entry.get("file", "")
        if not file.is_file():
            warnings.append(f"map {name!r} file missing ({entry.get('file')}); skipped")
            del maps[name]

    physical = doc.get("physical", {})
    physical.setdefault("radius_km", 69911.0)
    physical.setdefault("height_scale", 0.004)
    physical.setdefault("height_midlevel", 0.5)
    doc["physical"] = physical

    doc["_dir"] = str(folder)
    doc["_warnings"] = warnings
    return doc


def map_path(doc: dict, name: str) -> Path | None:
    entry = doc.get("maps", {}).get(name)
    if entry is None:
        return None
    return Path(doc["_dir"]) / entry["file"]


def ring_extent(doc: dict) -> tuple[float, float] | None:
    """(ring_inner_km, ring_outer_km) from the physical block, or None when the
    map set carries no ``rings`` map. Tolerant: a rings map without the physical
    radii falls back to Saturn's C..A span so the importer can still build the
    annulus."""
    if "rings" not in doc.get("maps", {}):
        return None
    physical = doc.get("physical", {})
    inner = physical.get("ring_inner_km", 74500.0)
    outer = physical.get("ring_outer_km", 136780.0)
    try:
        inner, outer = float(inner), float(outer)
    except (TypeError, ValueError):
        return (74500.0, 136780.0)
    if not (outer > inner > 0.0):
        return (74500.0, 136780.0)
    return (inner, outer)


def frames_block(doc: dict) -> dict | None:
    """The optional animation ``frames`` block (T7), or None for a still set.

    Tolerant: any non-mapping or empty value reads as "no sequence"."""
    frames = doc.get("frames")
    return frames if isinstance(frames, dict) and frames.get("files") else None


def frame_count(doc: dict) -> int:
    """Number of frames in the sequence (0 if this is a still map set)."""
    frames = frames_block(doc)
    if frames is None:
        return 0
    return int(frames.get("count", len(frames.get("files", []))))


def frame_zero_path(doc: dict, name: str) -> Path | None:
    """Absolute path to frame 0 of map ``name`` (``color`` uses ``frames.files``;
    other maps use ``frames.maps[name]``), or None when this map is not animated.

    Blender only needs the frame-0 file loaded with ``image.source='SEQUENCE'``;
    it discovers the rest from the ``_%04d`` numbering in the filename."""
    frames = frames_block(doc)
    if frames is None:
        return None
    if name == "color":
        files = frames.get("files") or []
    else:
        files = (frames.get("maps") or {}).get(name) or []
    if not files:
        return None
    return Path(doc["_dir"]) / files[0]

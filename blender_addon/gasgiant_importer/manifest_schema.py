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
    if version > SUPPORTED_SCHEMA:
        warnings.append(
            f"mapset schema_version {version} is newer than this add-on understands "
            f"({SUPPORTED_SCHEMA}); importing best-effort"
        )

    if doc.get("projection") != "equirectangular":
        raise MapsetError(f"unsupported projection {doc.get('projection')!r}")

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

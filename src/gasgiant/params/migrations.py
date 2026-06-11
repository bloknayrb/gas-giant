"""Preset format migrations.

Additive changes (new fields with defaults) need no migration. A breaking
change bumps CURRENT_PRESET_FORMAT and registers an upgrade function here;
upgrades are applied in sequence on load.
"""

from __future__ import annotations

from collections.abc import Callable

CURRENT_PRESET_FORMAT = 2


def _v1_to_v2(doc: dict) -> dict:
    """Format 2: appearance.palette (flat gradient) became
    appearance.palette_rows (latitude-anchored gradients). A single row at
    latitude 0 reproduces the v1 look exactly. Defensive: sparse presets may
    omit appearance or palette entirely."""
    appearance = doc.get("params", {}).get("appearance")
    if isinstance(appearance, dict) and "palette" in appearance:
        appearance["palette_rows"] = [
            {"latitude": 0.0, "stops": appearance.pop("palette")}
        ]
    return doc


# {from_version: upgrade(doc) -> doc at from_version + 1}
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {1: _v1_to_v2}


def migrate(doc: dict, from_format: int) -> dict:
    for version in range(from_format, CURRENT_PRESET_FORMAT):
        upgrade = _MIGRATIONS.get(version)
        if upgrade is None:
            raise ValueError(f"no migration registered for preset_format {version}")
        doc = upgrade(doc)
    return doc

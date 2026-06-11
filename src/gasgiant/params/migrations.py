"""Preset format migrations.

Additive changes (new fields with defaults) need no migration. A breaking
change bumps CURRENT_PRESET_FORMAT and registers an upgrade function here;
upgrades are applied in sequence on load.
"""

from __future__ import annotations

from collections.abc import Callable

CURRENT_PRESET_FORMAT = 1

# {from_version: upgrade(doc) -> doc at from_version + 1}
_MIGRATIONS: dict[int, Callable[[dict], dict]] = {}


def migrate(doc: dict, from_format: int) -> dict:
    for version in range(from_format, CURRENT_PRESET_FORMAT):
        upgrade = _MIGRATIONS.get(version)
        if upgrade is None:
            raise ValueError(f"no migration registered for preset_format {version}")
        doc = upgrade(doc)
    return doc

"""Parameter-diff -> invalidation tiers.

Compares two validated parameter trees and reports which tiers of work the
change requires. The engine dispatches the cheapest sufficient recompute:

- POST     re-derive maps from existing tracers (instant)
- VELOCITY rebuild the velocity field, sim continues under the new field
- RESTART  re-initialize the development run from step 0
"""

from __future__ import annotations

from pydantic import BaseModel

from gasgiant.params.model import PlanetParams, Tier


def diff_tiers(old: PlanetParams, new: PlanetParams) -> set[Tier]:
    """The set of tiers touched by changed fields (empty set: nothing changed)."""
    tiers: set[Tier] = set()
    _walk(type(old), old, new, tiers)
    return tiers


def _walk(model: type[BaseModel], old: BaseModel, new: BaseModel, tiers: set[Tier]) -> None:
    for name, info in model.model_fields.items():
        old_value = getattr(old, name)
        new_value = getattr(new, name)
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            _walk(ann, old_value, new_value, tiers)
            continue
        if old_value == new_value:
            continue
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        tier = extra.get("tier")
        if tier is not None:
            tiers.add(Tier(tier))


def needs_restart(tiers: set[Tier]) -> bool:
    return Tier.RESTART in tiers


def needs_velocity_rebuild(tiers: set[Tier]) -> bool:
    return Tier.VELOCITY in tiers or Tier.RESTART in tiers


def needs_repost(tiers: set[Tier]) -> bool:
    return bool(tiers)

"""Parameter-diff -> invalidation tiers (the tier-walk).

Compares two validated parameter trees and reports which tiers of work a change
requires. This lives in the ``params`` layer -- it imports only pydantic and
``params.model`` -- so params-layer consumers (e.g. the ramp validator in
``params.interp``) can use it without importing the ``engine`` layer.
``engine.invalidation`` re-exports ``diff_tiers`` so the many existing
``from gasgiant.engine.invalidation import diff_tiers`` call sites keep working.

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


def diff_tier_paths(old: PlanetParams, new: PlanetParams) -> list[tuple[str, Tier | None]]:
    """Every changed leaf as ``(dotted-path, tier)`` (tier is ``None`` for a leaf
    that carries no ``pfield`` tier metadata). Used by the ramp validator to name
    the offending fields in its error message; mirrors ``diff_tiers``'s walk
    (nested lists/optionals are compared as leaves, not descended into)."""
    paths: list[tuple[str, Tier | None]] = []
    _walk_paths(type(old), old, new, "", paths)
    return paths


def _walk_paths(
    model: type[BaseModel],
    old: BaseModel,
    new: BaseModel,
    prefix: str,
    paths: list[tuple[str, Tier | None]],
) -> None:
    for name, info in model.model_fields.items():
        old_value = getattr(old, name)
        new_value = getattr(new, name)
        path = f"{prefix}.{name}" if prefix else name
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            _walk_paths(ann, old_value, new_value, path, paths)
            continue
        if old_value == new_value:
            continue
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        tier_val = extra.get("tier")
        paths.append((path, Tier(tier_val) if tier_val is not None else None))


def needs_restart(tiers: set[Tier]) -> bool:
    return Tier.RESTART in tiers


def needs_velocity_rebuild(tiers: set[Tier]) -> bool:
    return Tier.VELOCITY in tiers or Tier.RESTART in tiers


def needs_repost(tiers: set[Tier]) -> bool:
    return bool(tiers)

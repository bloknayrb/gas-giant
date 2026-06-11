"""Seeded randomization of parameters from per-field range metadata.

Walks the model tree; every field whose metadata declares a ``rand`` range is
re-rolled from a generator derived from (seed, "randomize"). Fields can be
locked by dotted path (the GUI's lock toggles).
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel

from gasgiant.params.model import PlanetParams
from gasgiant.params.seeds import subseed


def randomize(
    seed: int, base: PlanetParams | None = None, locked: set[str] | None = None
) -> PlanetParams:
    base = base if base is not None else PlanetParams()
    locked = locked or set()
    rng = subseed(seed, "randomize")
    doc = base.model_dump()
    _walk(type(base), doc, rng, locked, prefix="")
    doc["seed"] = seed
    return PlanetParams.model_validate(doc)


def _walk(
    model: type[BaseModel],
    doc: dict[str, Any],
    rng: Any,
    locked: set[str],
    prefix: str,
) -> None:
    # Field order is the declaration order, which is stable -> deterministic draws.
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            _walk(annotation, doc[name], rng, locked, prefix=f"{path}.")
            continue
        rand = extra.get("rand")
        if rand is None:
            continue
        lo, hi = float(rand[0]), float(rand[1])
        if extra.get("log"):
            value = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            value = rng.uniform(lo, hi)
        if path in locked:
            continue  # draw consumed above so locking never shifts other fields
        if isinstance(doc[name], int) and not isinstance(doc[name], bool):
            doc[name] = int(round(value))
        else:
            doc[name] = value

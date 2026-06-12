"""Every tunable leaf in PlanetParams must map to a real widget.

This is the static guard against fields silently rendering as disabled text
(a tuple field without "color" in its name, a list shape the editor doesn't
know, ...). It imports panels for the pure leaf_kind classifier only — no
GUI is created.

Deliberate exception: "optional_model" (bands.template) is preset-only data
that intentionally renders as informational text — leaf_kind classifies it
explicitly, so it still counts as covered rather than silently falling
through.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from gasgiant.params.model import PlanetParams

panels = pytest.importorskip("gasgiant.app.panels")


def _walk(model: type[BaseModel], doc: dict, path: str = ""):
    for name, info in model.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            yield from _walk(ann, doc[name], f"{path}{name}.")
        else:
            yield f"{path}{name}", info, doc[name]


def test_every_leaf_renders_a_widget():
    params = PlanetParams()
    doc = params.model_dump()
    unrenderable = [
        path
        for path, info, value in _walk(type(params), doc)
        if panels.leaf_kind(path.rsplit(".", 1)[-1], info, value) is None
    ]
    assert not unrenderable, f"fields with no widget (disabled text): {unrenderable}"

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

from gasgiant.params.model import PlanetParams, SolverParams

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


def test_solver_params_all_fields_classified():
    """Every SolverParams field (the 5 new v1.6 knobs + type) must be
    classified by the panel walker: type→enum, poisson_iters→int,
    sor_omega/vort_hypervisc/coriolis_f0→float, vort_relax_tau→float(log).
    This guards against a new field silently falling through to disabled text."""
    sp = SolverParams()
    doc = sp.model_dump()
    unclassified = []
    for name, info in SolverParams.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            # Nested model (e.g. baroclinic) -> _draw_model renders it as a
            # collapsing section, not a leaf widget; its leaves are covered by
            # test_every_leaf_renders_a_widget. Not a fall-through to disabled text.
            continue
        kind = panels.leaf_kind(name, info, doc[name])
        if kind is None:
            unclassified.append(name)
    assert not unclassified, (
        f"SolverParams fields with no widget (would render as disabled text): {unclassified}"
    )

    # Also assert the specific widget kinds the plan specifies.
    expected_kinds = {
        "type": "enum",
        "poisson_iters": "int",
        "sor_omega": "float",
        "vort_hypervisc": "float",
        "coriolis_f0": "float",
        "vort_relax_tau": "float",
    }
    for field, expected in expected_kinds.items():
        info = SolverParams.model_fields[field]
        got = panels.leaf_kind(field, info, doc[field])
        assert got == expected, (
            f"SolverParams.{field}: expected widget kind '{expected}', got '{got}'"
        )

    # vort_relax_tau must have log=True in its schema extra (log slider).
    tau_info = SolverParams.model_fields["vort_relax_tau"]
    extra = tau_info.json_schema_extra if isinstance(tau_info.json_schema_extra, dict) else {}
    assert extra.get("log") is True, (
        "vort_relax_tau must carry log=True in json_schema_extra for the log slider"
    )

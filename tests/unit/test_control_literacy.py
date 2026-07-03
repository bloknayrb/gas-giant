"""B2 control-literacy pass (W11).

Pins the reviewed tooltip rewrites so they can't silently regress to
jargon-first copy:

- B2-2: the worst-10 solver/waves tooltips now LEAD with the visual read and
  keep the physics parenthesized.
- B2-1: the two long descriptions (deformation_radius, vort_psi_drag) are
  front-loaded -- picture first, physics after.
- B2-3: the baroclinic cadence trio is labeled ("Fixed cadence") instead of
  rendering unlabeled with ui="".
- B2-4: radian-quantified fields carry a degree gloss; the Help glossary
  covers GRS/anticyclone/cyclonic/retrograde; every EmissionParams tooltip
  warns where emission is (and is not) previewable.
"""

from __future__ import annotations

import pytest

from gasgiant.params.model import (
    BaroclinicParams,
    EmissionParams,
    PlanetParams,
    SolverParams,
    WavesParams,
    field_meta,
)


def _desc(model: type, field: str) -> str:
    return model.model_fields[field].description or ""


# -- B2-2: visual read first, physics parenthesized ------------------------------


@pytest.mark.parametrize(
    ("field", "lead", "kept_physics"),
    [
        ("type", "How clouds move", "v1.5"),
        ("poisson_iters", "Solver accuracy per step", "SOR"),
        ("sor_omega", "Solver convergence speed", "over-relaxation"),
        ("vort_relax_tau", "How tightly the flow is leashed", "timescale"),
        ("vort_hypervisc", "Fine-scale smoothing", "hyperviscosity"),
        ("coriolis_f0", "Planet-rotation strength", "Rhines"),
        ("vort_inject_scale", "Size of the injected churn", "detail_freq"),
        ("vort_drag", "Global brake on swirling", "Rayleigh"),
    ],
)
def test_solver_tooltips_lead_with_the_picture(field, lead, kept_physics):
    desc = _desc(SolverParams, field)
    assert desc.startswith(lead), f"{field} must lead with the visual read: {desc!r}"
    assert kept_physics in desc, f"{field} must keep the physics term for power users"


def test_vort_drag_points_at_the_better_lever():
    assert "vort_psi_drag" in _desc(SolverParams, "vort_drag")


def test_festoon_wavenumber_tooltip_is_plain_language():
    desc = _desc(WavesParams, "festoon_wavenumber")
    assert desc.startswith("How many festoon plumes")
    assert "Rossby" in desc, "physics term stays, parenthesized"


# -- B2-1: the two long descriptions are front-loaded -----------------------------


def test_deformation_radius_front_loaded():
    desc = _desc(SolverParams, "deformation_radius")
    assert desc.startswith("Storm locality"), "picture first"
    assert desc.index("locality") < desc.index("Rossby"), "physics after"


def test_vort_psi_drag_front_loaded():
    desc = _desc(SolverParams, "vort_psi_drag")
    assert desc.index("festoons") < desc.index("hypofriction"), "picture first"
    assert "0 = off" in desc


# -- B2-3: cadence trio labeled -----------------------------------------------------


@pytest.mark.parametrize("field", ["warmup_steps", "baro_steps_per_update", "update_every"])
def test_cadence_trio_labeled_and_advised(field):
    meta = field_meta(BaroclinicParams, field)
    assert meta.ui == "Fixed cadence", "renders under a sub-label, not unlabeled"
    assert meta.adv is True
    desc = _desc(BaroclinicParams, field)
    assert "leave at default" in desc, "the tooltip advises against fiddling"
    assert meta.rand is None


# -- B2-4: degree glosses, glossary, emission preview warnings --------------------


def test_radian_fields_carry_degree_gloss():
    radian_fields = []
    for name, info in PlanetParams.model_fields.items():
        ann = info.annotation
        if not (isinstance(ann, type) and hasattr(ann, "model_fields")):
            continue
        for leaf, leaf_info in ann.model_fields.items():
            desc = leaf_info.description or ""
            if "radians of" in desc.lower():
                radian_fields.append((f"{name}.{leaf}", desc))
    assert radian_fields, "sanity: the radian-quantified fields still exist"
    for path, desc in radian_fields:
        assert "57.3" in desc, f"{path} needs the 1 rad = 57.3 deg gloss: {desc!r}"


def test_help_glossary_covers_the_unglossed_vocabulary():
    main = pytest.importorskip("gasgiant.app.main")
    text = main._HELP_GLOSSARY
    for term in ("Great Red Spot", "Anticyclone", "cyclonic", "Retrograde", "57.3"):
        assert term in text, f"glossary must cover {term!r}"


def test_every_emission_tooltip_warns_about_preview_visibility():
    for name, info in EmissionParams.model_fields.items():
        desc = info.description or ""
        assert "Emission channel" in desc, f"emission.{name} lacks the preview note"
        assert "Color" in desc, f"emission.{name} must name where it is NOT visible"

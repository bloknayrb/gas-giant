"""Authored widget captions (``pfield(label=...)``).

The caption is the only text an artist reads while SCANNING the panel -- the
description appears on hover. Derived from the field name, a solver lever reads
``vort psi drag``; authored, it reads "Swirl brake (large only)".

The risk this file guards is not a wrong caption, it is a caption that quietly
breaks something else: search, the docs, or an imgui id.
"""

from __future__ import annotations

import re
import typing

import pytest
from pydantic import BaseModel

from gasgiant.params.model import (
    FieldMeta,
    PlanetParams,
    StormOverride,
    StormsParams,
    derived_label,
    field_label,
)

# NOT a module-level importorskip: only two tests below need the GUI layer, and
# skipping the whole file would take the caption-hygiene guards with it in
# exactly the GUI-extra-free environment field_label's own docstring describes.


def _panels():
    return pytest.importorskip("gasgiant.app.panels")


def _leaves():
    """``(dotted_path, field_name, FieldInfo)`` for every pfield leaf."""
    out, seen = [], set()

    def walk(model, prefix=""):
        for name, info in model.model_fields.items():
            for member in (info.annotation, *typing.get_args(info.annotation)):
                if isinstance(member, type) and issubclass(member, BaseModel):
                    walk(member, f"{prefix}{name}.")
            if "tier" in (info.json_schema_extra or {}):
                key = (model.__name__, name)
                if key not in seen:
                    seen.add(key)
                    out.append((f"{prefix}{name}", name, info))

    walk(PlanetParams)
    return out


def _labelled():
    return [(p, n, i) for p, n, i in _leaves() if FieldMeta.of(i).label]


def test_some_fields_are_labelled():
    """Sanity: the rest of this file is vacuous if the metadata stopped landing."""
    assert len(_labelled()) >= 20


def test_unlabelled_fields_fall_back_to_the_derived_caption():
    for _path, name, info in _leaves():
        if not FieldMeta.of(info).label:
            assert field_label(name, info) == derived_label(name)


def test_authored_captions_are_plain_text():
    """Second net under ``pfield``'s own import-time validation -- that is the
    guard that cannot be skipped, this one reports every offender at once.

    ``##`` is imgui's id separator and silently hides everything after it. A
    lone ``#`` and ``|`` are harmless in the ``### {label}`` heading the doc
    generator emits today; they are banned pre-emptively so a caption stays
    safe the day it lands somewhere stricter.
    """
    for path, name, info in _labelled():
        label = field_label(name, info)
        assert label.strip() == label, f"{path}: padded caption {label!r}"
        assert "#" not in label, f"{path}: '#' is imgui id syntax"
        assert "|" not in label, f"{path}: reserved for markdown tables"
        assert "\n" not in label, f"{path}: captions are single-line"


def test_authored_captions_drop_the_engine_vocabulary():
    """A caption that still says 'vort' has not done its job -- that is the
    exact string the artist could not read in the first place.

    Word-boundary matched: a bare substring ban rejects legitimate future
    captions ('tau' is inside 'Plateau', 'psi' inside 'Ellipsis').
    """
    banned = ("vort", "psi", "sor", "coriolis", "poisson", "hypervisc", "wavenumber",
              "tau", "flat", "qgpv", "rossby", "advect")
    for path, name, info in _labelled():
        low = field_label(name, info).lower()
        hits = [t for t in banned if re.search(rf"\b{t}\b", low)]
        assert not hits, f"{path}: caption still jargon {hits}: {low!r}"


def test_authored_captions_are_unique():
    """Two fields sharing a caption are indistinguishable in the panel and
    produce duplicate ``###`` anchors in the generated doc. (Widget ids are
    safe regardless -- panels pushes the field NAME -- so this is a
    readability guard, not a correctness one.)"""
    seen: dict[str, str] = {}
    for path, name, info in _labelled():
        label = field_label(name, info)
        assert label not in seen, f"{path} and {seen[label]} share the caption {label!r}"
        seen[label] = path


def test_shared_field_names_keep_one_caption():
    """``StormsParams`` and ``StormOverride`` declare the same concepts, and
    the cast editor draws them beside the section leaves. Labelling one and not
    the other would put two captions on one concept in the same panel -- and
    the doc generator never enumerates StormOverride, so a cast-only label
    would show in the GUI and never in the docs."""
    for name in set(StormsParams.model_fields) & set(StormOverride.model_fields):
        assert field_label(name, StormsParams.model_fields[name]) == field_label(
            name, StormOverride.model_fields[name]
        ), f"{name}: section leaf and cast row disagree on the caption"


@pytest.mark.parametrize("case", _labelled(), ids=lambda c: c[0])
def test_relabelled_fields_stay_findable_by_their_old_caption(case):
    """The regression an authored label invites: the haystack carries the shown
    caption, so REPLACING the derived form un-finds the relabelled field.
    Searching "vort psi" must still reach vort_psi_drag.

    Genuinely parametrized, so a break reports every affected field rather than
    aborting on the first. Mutation-checked: dropping the derived form fails 21
    of these 22 while test_panel_state.py stays green -- its four search tests
    exercise width_jitter, hero_solid_core, gamma and seed, none of which carry
    an authored label today.
    """
    panels = _panels()
    path, name, info = case
    state = panels.PanelState(show_advanced=True, search=derived_label(name))
    assert panels._leaf_visible(name, info, {}, state), (
        f"{path} is no longer findable by its derived caption {state.search!r}"
    )


@pytest.mark.parametrize("case", _labelled(), ids=lambda c: c[0])
def test_relabelled_fields_are_findable_by_the_new_caption_too(case):
    """The other half: the caption an artist can actually see must find it."""
    panels = _panels()
    path, name, info = case
    state = panels.PanelState(show_advanced=True, search=field_label(name, info))
    assert panels._leaf_visible(name, info, {}, state), f"{path} not findable by its caption"


def test_relabelled_fields_show_their_name_in_the_tooltip():
    """The caption was the only place the field NAME appeared in the GUI.
    Descriptions cross-reference fields by name ("prefer vort_psi_drag") and
    validator errors surface through toasts naming them ("sor_omega=2.0 must
    be strictly in (1.0, 2.0)") -- so without the name in the tooltip, dragging
    "Solver convergence speed" to its max reports an error about a string that
    appears nowhere on screen."""
    panels = _panels()
    for path, name, info in _labelled():
        tip = panels._leaf_tip(name, info)
        assert name in tip, f"{path}: tooltip never names the field"
        assert (info.description or "") in tip, f"{path}: tooltip lost its description"


def test_unlabelled_fields_get_a_bare_description_tooltip():
    """No prefix where the caption already IS the name -- that would be noise
    on 177 of 199 fields."""
    panels = _panels()
    for _path, name, info in _leaves():
        if not FieldMeta.of(info).label:
            assert panels._leaf_tip(name, info) == (info.description or "")


# -- the invariant the params-layer placement exists to protect ----------------

_WIDGETS = (
    "slider_float", "slider_int", "combo", "checkbox",
    "input_int", "input_float", "drag_float", "color_edit3",
)


def _captions_drawn(panels, imgui, search):
    """Every caption the panel actually hands to imgui for ``search``."""
    seen: list[str] = []

    def spy(real):
        def wrapper(label, *a, **k):
            seen.append(label)
            return real(label, *a, **k)
        return wrapper

    originals = {n: getattr(imgui, n) for n in _WIDGETS if hasattr(imgui, n)}
    for n, real in originals.items():
        setattr(imgui, n, spy(real))
    try:
        state = panels.PanelState(show_advanced=True, search=search)
        imgui.new_frame()
        imgui.begin("host", None, 0)
        panels.draw_params_panel(PlanetParams(), state)
        imgui.end()
        imgui.end_frame()
    finally:
        for n, real in originals.items():
            setattr(imgui, n, real)
    return seen


@pytest.mark.parametrize("case", _labelled(), ids=lambda c: c[0])
def test_the_panel_draws_the_authored_caption(case):
    """Kills the mutant the doc gate structurally cannot see: reverting a
    panels caption to the derived form while the generator keeps the authored
    one. Both call ``field_label``; nothing but this asserts the GUI half.
    """
    panels = _panels()
    imgui = pytest.importorskip("imgui_bundle.imgui")
    _path, name, info = case
    caption = field_label(name, info)
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(900.0, 700.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    try:
        drawn = _captions_drawn(panels, imgui, caption)
    finally:
        imgui.destroy_context(ctx)
    assert any(c.startswith(caption) for c in drawn), (
        f"{name}: panel drew {[c for c in drawn if name.split('_')[0] in c.lower()]!r}, "
        f"not the authored caption {caption!r}"
    )


@pytest.mark.parametrize("case", _labelled(), ids=lambda c: c[0])
def test_the_doc_renders_the_authored_caption(case):
    """The other half of the same invariant, asserted against the MODEL rather
    than against regenerated output. ``--check`` only reports staleness: revert
    a generator site, regenerate, and it goes green on a doc that disagrees
    with the app. This does not.
    """
    import pathlib

    _path, name, info = case
    doc = pathlib.Path(__file__).resolve().parents[2] / "docs" / "sliders.md"
    headings = set(re.findall(r"^### (.+)$", doc.read_text(encoding="utf-8"), re.M))
    assert field_label(name, info) in headings, (
        f"{name}: docs/sliders.md has no heading for its authored caption "
        f"(regenerate with render_slider_examples.py --no-render)"
    )


# The captions themselves, pinned. Without this a label can be deleted,
# reworded or replaced with garbage and every other test here still passes --
# the jargon guard only fires when the DERIVED form happens to contain a banned
# token, so it is structurally blind to e.g. dt_scale ("dt scale").
EXPECTED_LABELS = {
    "sim.dt_scale": "Time step",
    "solver.poisson_iters": "Solver accuracy",
    "solver.sor_omega": "Solver convergence speed",
    "solver.deformation_radius": "Storm reach (0 = unlimited)",
    "solver.vort_relax_tau": "Flow leash",
    "solver.vort_hypervisc": "Fine smoothing",
    "solver.coriolis_f0": "Rotation strength",
    "solver.vort_inject": "Churn strength",
    "solver.vort_inject_scale": "Churn scale",
    "solver.vort_inject_mask": "Churn placement",
    "solver.vort_drag": "Swirl brake (all scales)",
    "solver.vort_eddy_drag": "Eddy brake (all scales, jets spared)",
    "solver.vort_psi_drag": "Swirl brake (large only)",
    "bands.warp_freq": "Band meander scale",
    "bands.detail_freq": "Band detail scale",
    "turbulence.relax_tau": "Turbulence leash",
    "turbulence.kh_amplitude": "Billow strength",
    "turbulence.kh_wavenumber": "Billow count",
    "waves.festoon_wavenumber": "Festoon count",
    "waves.ribbon_wavenumber": "Ribbon wave count",
    "waves.festoon_hero_wavenumber": "Festoon count (hero)",
    "detail.cirrus_fiber_freq": "Cirrus fiber scale",
}


def test_authored_captions_match_the_pin():
    assert {p: field_label(n, i) for p, n, i in _labelled()} == EXPECTED_LABELS

"""Auto-generated parameter panels.

Widgets are derived from the pydantic model: nested models become collapsing
headers, numeric fields become sliders bounded by their validation constraints,
color tuples become color pickers, gradient-stop lists get a simple stops
editor. Adding a sim parameter in params/model.py is all it takes to get UI.

Edits mutate a plain dict draft; the caller validates and commits. The engine
never sees a mid-drag invalid state.

Phase 3 layers panel-navigation affordances on top of that reflection: a
search/filter box, per-field reset/lock/modified markers, tier-cost badges, and
section help tooltips. All of it is driven by ``PanelState`` -- GUI-adjacent
state owned by the caller (``StudioApp``), not ``gasgiant.params`` (the layer
contract forbids params importing imgui/app).

Phase 4 adds a Basic/Advanced split (an ``Advanced`` checkbox next to search;
``pfield(adv=True)`` leaves hide in Basic unless a search matches them) and
sub-grouping within a section via refined ``ui`` labels (``separator_text`` on
a label change between consecutive leaves).
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
from typing import Any

import annotated_types
from imgui_bundle import imgui
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from gasgiant.params.model import FieldMeta, PlanetParams, hero_latitude_cap


def _bounds(info: FieldInfo) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in info.metadata:
        if isinstance(m, annotated_types.Ge):
            lo = float(m.ge)
        elif isinstance(m, annotated_types.Le):
            hi = float(m.le)
    return lo, hi


def _is_color_field(name: str, value: Any) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
        and "color" in name
    )


@dataclasses.dataclass
class PanelState:
    """GUI-adjacent panel-navigation state, owned by ``StudioApp`` and
    threaded through ``draw_params_panel``/``_draw_model``/``_draw_leaf``.
    Lives here (not ``gasgiant.params``) because it carries no engine-relevant
    data -- the layer contract forbids params importing imgui/app.

    - ``search`` -- the active filter substring (case-insensitive).
    - ``show_advanced`` -- Basic/Advanced split (Phase 4): False (the
      default -- newcomers land in Basic) hides any leaf whose ``pfield``
      metadata carries ``adv=True``, unless a search is active (a search
      always overrides the gate, so an advanced field stays findable by
      name even in Basic mode).
    - ``locked`` -- dotted field paths excluded from ``randomize()`` (the
      panel's right-click "Lock for randomize" toggle and the header seed
      lock both write here).
    - ``focus_search_requested`` -- Phase 7's ``/`` shortcut sets this
      (``StudioApp._handle_shortcuts``); ``_draw_search_box`` consumes it the
      next time it draws the search box, via ``imgui.set_keyboard_focus_here()``
      called immediately before the input widget, then clears the flag so the
      focus request fires exactly once.
    """

    search: str = ""
    show_advanced: bool = False
    locked: set[str] = dataclasses.field(default_factory=set)
    focus_search_requested: bool = False


_DEFAULTS_BASELINE: dict[str, Any] | None = None


def _defaults_baseline() -> dict[str, Any]:
    """One static ``PlanetParams().model_dump()`` baseline, computed once and
    cached at module scope. Used for both the modified-marker and the
    right-click "Reset to default" value -- never per-field ``get_default()``
    calls, never recomputed per frame."""
    global _DEFAULTS_BASELINE
    if _DEFAULTS_BASELINE is None:
        _DEFAULTS_BASELINE = PlanetParams().model_dump()
    return _DEFAULTS_BASELINE


# One-line blurbs for the header (?) markers, keyed by the top-level
# PlanetParams field name. Not exhaustive of every nested section -- only the
# direct top-level sections, per the Phase 3 spec.
_SECTION_BLURBS: dict[str, str] = {
    "sim": "Grid resolution and development-run step budget.",
    "solver": "Velocity-field solver: kinematic vs. vorticity, Poisson/SOR "
    "tuning, drag and injection.",
    "bands": "Latitudinal band layout and base color banding.",
    "jets": "Zonal jet speed/shear profile riding on the bands.",
    "turbulence": "Eddy/noise turbulence layered onto the velocity field.",
    "storms": "Discrete storms, including the hero (Great-Red-Spot-style) vortex.",
    "waves": "Rossby/gravity-wave ripple overlays.",
    "poles": "Polar vortex style and cap appearance.",
    "appearance": "Color palette, contrast, and tonal-mapping controls.",
    "detail": "Fine-scale procedural texture detail layers.",
    "emission": "Self-emissive glow channels (lightning, aurora, hot spots). "
    "Preview via the viewport's Emission channel (aurora included) — the "
    "Color preview does not composite emission. Exported to emission.exr.",
    "physical": "Planet radius and physical-shading parameters.",
    "export": "Output map resolution and PNG compression for Export.",
}


def draw_params_panel(
    params: PlanetParams, state: PanelState | None = None
) -> tuple[dict[str, Any], bool, bool]:
    """Draw all parameter widgets against a fresh draft of ``params``.

    ``state`` is the Phase 3 navigation state (search/locked/show_advanced).
    It defaults to a throwaway ``PanelState()`` so existing callers (and any
    test not yet updated) keep working unchanged.

    Returns ``(draft, any_changed, any_committed)``:

    - ``any_changed``  — a value changed this frame (mid-drag included).
    - ``any_committed`` — a widget *finished* editing this frame (slider
      release, combo/checkbox/color pick, ``input_int`` Enter/focus-loss, or a
      structural palette mutation, or a right-click Reset-to-default). The
      caller defers heavy (velocity/restart) rebuilds to ``any_committed``
      frames so a drag commits once on release instead of every frame.
    """
    if state is None:
        state = PanelState()
    _draw_search_box(state)
    draft = params.model_dump()
    changed, committed = _draw_model(
        type(params), draft, _defaults_baseline(), state, top_level=True
    )
    return draft, changed, committed


def _draw_search_box(state: PanelState) -> None:
    imgui.set_next_item_width(-140.0)
    if state.focus_search_requested:
        # Must be called immediately before the widget it targets -- ordering
        # matters here (imgui associates the focus request with the very next
        # item). Cleared right away so the request fires once, not every frame.
        imgui.set_keyboard_focus_here()
        state.focus_search_requested = False
    changed, text = imgui.input_text_with_hint("##panel_search", "search fields...", state.search)
    if changed:
        state.search = text
    imgui.same_line()
    if imgui.button("X##clear_search"):
        state.search = ""
    imgui.same_line()
    _, state.show_advanced = imgui.checkbox("Advanced", state.show_advanced)
    imgui.separator()


def _advanced_visible(info: FieldInfo, state: PanelState) -> bool:
    """The Basic/Advanced half of leaf visibility: a leaf not marked
    ``adv=True`` is always visible; an ``adv=True`` leaf is visible only
    while ``state.show_advanced`` is on. Search overrides this entirely --
    see ``_leaf_visible``, the only caller."""
    if not FieldMeta.of(info).adv:
        return True
    return state.show_advanced


def _leaf_visible(name: str, info: FieldInfo, doc: dict[str, Any], state: PanelState) -> bool:
    """Combined search + Basic/Advanced visibility predicate. Shared verbatim
    by the leaf draw (skip drawing a non-matching/hidden leaf) AND the
    section pre-pass (skip a section with zero visible leaves) -- the SAME
    function, not two copies that could drift.

    No active search: gate purely on Basic/Advanced (``_advanced_visible``).
    Active search: match the field name, its display label, and its
    description, case-insensitive substring -- and the match ALONE decides
    visibility, bypassing the Basic/Advanced gate entirely, so a
    searched-for advanced field is still findable with Advanced off.
    ``doc`` isn't used by the match itself (kept in the signature for
    symmetry with the draw call site / future filters).
    """
    query = state.search.strip().lower()
    if not query:
        return _advanced_visible(info, state)
    label = name.replace("_", " ")
    haystack = f"{name} {label} {info.description or ''}".lower()
    return query in haystack


def _subtree_has_match(model: type[BaseModel], doc: dict[str, Any], state: PanelState) -> bool:
    """Section pre-pass: does this (possibly nested) section contain at least
    one leaf matching the active search? Recurses into nested models and
    calls ``_leaf_visible`` for every leaf -- the same predicate the leaf draw
    itself gates on."""
    for name, info in model.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            if _subtree_has_match(ann, doc[name], state):
                return True
            continue
        if _leaf_visible(name, info, doc, state):
            return True
    return False


def _leaf_changed(current: Any, default: Any) -> bool:
    """True if a leaf value differs from its default, treating a color list and
    the tuple a live color edit produces as equal. ``model_dump()`` emits colors
    as lists, but ``color_edit3`` writes back ``tuple(rgb)``, so a plain ``!=``
    reads a color dragged back to its default as still modified (tuple != list) --
    leaving the ``*`` marker stuck and inflating the "N advanced differ" count."""
    if isinstance(current, (list, tuple)) and isinstance(default, (list, tuple)):
        return list(current) != list(default)
    return current != default


def _count_differs_from_default(
    model: type[BaseModel], doc: dict[str, Any], baseline: dict[str, Any]
) -> int:
    """Recursive count of leaves in this (possibly nested) section whose
    current value differs from the static defaults baseline. Used by the
    hidden-advanced-settings hint -- reuses ``_defaults_baseline()``, never
    recomputes a default per field (Phase 3's rule for the modified-marker/
    reset-to-default machinery, reused verbatim here)."""
    count = 0
    for name, info in model.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            count += _count_differs_from_default(ann, doc[name], baseline[name])
            continue
        if name in baseline and _leaf_changed(doc[name], baseline[name]):
            count += 1
    return count


def _draw_hidden_advanced_hint(
    model: type[BaseModel], doc: dict[str, Any], baseline: dict[str, Any]
) -> None:
    """H4 + Round 2 MED-3: in Basic mode a fully-advanced section (Solver/
    Emission/Physical/Baroclinic, or any other section that happens to have
    zero Basic-visible leaves) renders zero leaves, so a preset that changed
    one of them silently "disappears" from view. Drawn right under the
    section header regardless of open/closed state -- the caller only
    invokes this when the section currently has zero visible leaves, so
    every leaf counted here IS hidden."""
    n = _count_differs_from_default(model, doc, baseline)
    if n <= 0:
        return
    plural = "s" if n != 1 else ""
    imgui.text_colored(
        imgui.ImVec4(*_MODIFIED_COLOR),
        f"{n} advanced setting{plural} differ from default -- toggle Advanced to edit",
    )


def _draw_bands_template_escape(bands_doc: dict[str, Any]) -> tuple[bool, bool]:
    """Basic-visible escape for ``bands.template`` (an ``adv=True`` field,
    hidden in Basic mode): when a preset sets a template, seeded value
    seasoning (value_contrast, hue_jitter, the width knobs) goes inert, so a
    Basic-mode user dragging those sliders and seeing nothing happen would
    otherwise have no clue why. Banner + "Clear template" button, drawn
    whenever a template is set regardless of Advanced/collapsed state
    (H4 + Round 2 MED-3). Returns a synthetic (changed, committed) pair on
    click, same as the composite editors' add/remove-row buttons."""
    if bands_doc.get("template") is None:
        return False, False
    imgui.text_colored(
        imgui.ImVec4(*_MODIFIED_COLOR),
        "band template is set -- overrides the band sliders below",
    )
    imgui.same_line()
    # B4-2: clearing is destructive (the template values are kept nowhere but
    # the undo history), and the startup preset ships WITH a template -- so
    # the button stages a confirm modal instead of clearing outright.
    title = "Clear band template?"
    if imgui.small_button("Clear template"):
        imgui.open_popup(title)
    changed = committed = False
    center = imgui.get_main_viewport().get_center()
    imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
    if imgui.begin_popup_modal(title, None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text_wrapped(_CLEAR_TEMPLATE_CONFIRM)
        imgui.separator()
        if imgui.button("Clear##template"):
            bands_doc["template"] = None
            changed = committed = True
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel##template"):
            imgui.close_current_popup()
        imgui.end_popup()
    return changed, committed


# B4-2: the confirm-modal copy, a module constant so tests can pin it.
_CLEAR_TEMPLATE_CONFIRM = (
    "Clear the preset's calibrated band skeleton? The seeded band sliders "
    "take over, and the template values are not kept anywhere -- only Undo "
    "(Ctrl+Z) brings them back."
)


def _draw_hero_latitude_escape(storms_doc: dict[str, Any]) -> tuple[bool, bool]:
    """Basic-visible unpin escape for ``storms.hero_latitude`` (an
    ``adv=True``, preset-only field): a preset can pin the hero storm's
    latitude, which is otherwise only reachable in Advanced mode. Unpinning
    (back to None -- seeded placement) always satisfies
    ``_validate_hero_latitude`` (it only checks a non-None value), so no
    bounds juggling is needed here -- just a plain reset-to-None."""
    lat = storms_doc.get("hero_latitude")
    if lat is None:
        return False, False
    imgui.text_colored(
        imgui.ImVec4(*_MODIFIED_COLOR), f"hero latitude pinned to {lat:.1f} deg"
    )
    imgui.same_line()
    if imgui.small_button("Unpin latitude"):
        storms_doc["hero_latitude"] = None
        return True, True
    return False, False


# B1-7/B4-3: aurora writes the exported emission map's alpha channel; the
# viewport composites it (alpha x aurora color) into the Emission channel
# preview, but nothing in the Color preview moves when it is enabled. Said
# exactly where the aurora controls live, at the moment it applies (aurora is
# on), so the zero-feedback Color-view slider drag stops reading as "broken".
_AURORA_PREVIEW_NOTE = (
    "aurora is on — preview it in the viewport's Emission channel; "
    "it is not composited into the Color preview"
)


def _draw_emission_aurora_note(emission_doc: dict[str, Any]) -> None:
    """Informational banner under the Emission header while aurora is active
    (same always-visible idiom as ``_draw_bands_template_escape``, minus the
    button: there is nothing to fix, only something to know). Draws regardless
    of Advanced/collapsed state -- the confusion happens precisely when the
    (advanced) aurora fields are hidden."""
    if emission_doc.get("aurora_strength", 0.0) <= 0.0:
        return
    imgui.text_colored(imgui.ImVec4(*_MODIFIED_COLOR), _AURORA_PREVIEW_NOTE)


def _draw_help_marker(text: str) -> None:
    imgui.text_disabled("(?)")
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


def _section_header(label: str, flags: int, searching: bool) -> bool:
    """Draw a collapsing header, forced open every frame while a search is
    active.

    While ``searching``, the forced ``SetNextItemOpen`` write happens under a
    SEPARATE ``push_id("search")`` id-stack scope, wrapping only this call --
    not the section's content. That puts the forced-open write in a different
    imgui storage slot than the header's normal (non-search) open/closed
    state, so clearing the search and drawing the header again under its
    normal id reads back whatever the user had manually toggled (guard test:
    search type-then-clear preserves header open/closed state). Verified
    empirically with a headless imgui context: priming the normal-scope state
    closed, forcing the search-scope state open for several frames, then
    clearing the search reads the normal scope back as still closed.
    """
    if searching:
        imgui.push_id("search")
        imgui.set_next_item_open(True, imgui.Cond_.always)
    opened = imgui.collapsing_header(label, flags)
    if searching:
        imgui.pop_id()
    return opened


def _draw_model(
    model: type[BaseModel],
    doc: dict[str, Any],
    baseline: dict[str, Any],
    state: PanelState,
    top_level: bool = False,
    prefix: str = "",
) -> tuple[bool, bool]:
    changed = False
    committed = False
    searching = bool(state.search.strip())
    prev_ui: str | None = None  # sub-group separator tracking (leaves only)
    for name, info in model.model_fields.items():
        ann = info.annotation
        path = f"{prefix}{name}"
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            # has_match combines search-match AND the Basic/Advanced gate
            # (both live in _leaf_visible, which _subtree_has_match calls per
            # leaf) -- one predicate, two uses below, never two copies that
            # could drift (guard test M9).
            has_match = _subtree_has_match(ann, doc[name], state)
            # Zero-match section suppression is search-only: outside a search
            # every section header still renders (a fully-advanced section
            # like Solver must show its header in plain browsing).
            if searching and not has_match:
                continue
            flags = imgui.TreeNodeFlags_.default_open if top_level else 0
            opened = _section_header(name.capitalize(), flags, searching)
            if top_level:
                blurb = _SECTION_BLURBS.get(name)
                if blurb:
                    imgui.same_line()
                    _draw_help_marker(blurb)
            if not searching and not has_match:
                # Not searching, and the Basic/Advanced gate hid every leaf
                # in this section (Solver/Emission/Physical/Baroclinic and
                # any other section that happens to have zero Basic leaves):
                # surface that a preset may have changed one of them anyway.
                _draw_hidden_advanced_hint(ann, doc[name], baseline[name])
            if name == "bands":
                c, cm = _draw_bands_template_escape(doc[name])
                changed |= c
                committed |= cm
            if name == "storms":
                c, cm = _draw_hero_latitude_escape(doc[name])
                changed |= c
                committed |= cm
            if name == "emission":
                _draw_emission_aurora_note(doc[name])
            if opened:
                imgui.push_id(name)
                imgui.indent(8.0)
                c, cm = _draw_model(ann, doc[name], baseline[name], state, prefix=f"{path}.")
                changed |= c
                committed |= cm
                imgui.unindent(8.0)
                imgui.pop_id()
            continue
        if _leaf_visible(name, info, doc, state):
            # Sub-group separator: emit only on a CHANGE from the previous
            # VISIBLE leaf's ui label, and only when that label is truthy --
            # a section with one constant ui value (every section besides
            # Storms today) never differs from its own previous value, so it
            # renders zero separators, byte-for-byte unchanged. Falsy ui is
            # transparent to grouping: it neither triggers nor absorbs a
            # boundary. (Baroclinic's cadence trio now carries ui="Fixed
            # cadence" -- B2-3 -- so it draws under its own sub-label.)
            ui = FieldMeta.of(info).ui or None
            if ui and prev_ui is not None and ui != prev_ui:
                imgui.separator_text(ui)
            if ui:
                prev_ui = ui
        c, cm = _draw_leaf(name, info, doc, baseline, state, path)
        changed |= c
        committed |= cm
    return changed, committed


def leaf_kind(name: str, info: FieldInfo, value: Any) -> str | None:
    """Which widget _draw_leaf renders for this field, or None if it has no
    widget. Kept as a pure function so a static test can assert every leaf
    in PlanetParams is renderable without opening a GUI."""
    import types
    from enum import StrEnum
    from typing import Union, get_args, get_origin

    ann = info.annotation
    if get_origin(ann) in (Union, types.UnionType):
        inner = [a for a in get_args(ann) if a is not type(None)]
        if len(inner) == 1 and isinstance(inner[0], type) and issubclass(inner[0], BaseModel):
            # Optional nested model (bands.template): preset-only data, shown
            # as informational text -- _draw_leaf's fallback branch.
            return "optional_model"
        if len(inner) == 1 and inner[0] is float:
            return "optional_float"
        if len(inner) == 1 and inner[0] is int:
            return "optional_int"
        if len(inner) == 1 and inner[0] is str:
            # Optional string (mask.file): a text-entry + Browse button; None =
            # unset. Placed beside optional_float/int so str|None doesn't fall
            # through to None (which fails test_panels_coverage).
            return "optional_str"
    if isinstance(ann, type) and issubclass(ann, StrEnum):
        return "enum"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if _is_color_field(name, value):
        return "color"
    if isinstance(value, str):
        return "str"
    # Annotation-keyed list-of-model editor (storms.cast). Placed BEFORE the
    # value-based list checks so the EMPTY default ([], no value[0] to sniff)
    # still classifies. The palette/stops lists are also list[BaseModel] but
    # have bespoke value-shape editors below, so they are excluded here.
    if get_origin(ann) is list:
        args = get_args(ann)
        if len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            from gasgiant.params.model import GradientStop, PaletteRow
            if args[0] not in (GradientStop, PaletteRow):
                return "model_list"
    if isinstance(value, list) and value and isinstance(value[0], dict) and "pos" in value[0]:
        return "stops"
    if isinstance(value, list) and value and isinstance(value[0], dict) and "stops" in value[0]:
        return "palette_rows"
    return None


# Widget kinds that draw a single imgui item whose end-of-edit is reported by
# is_item_deactivated_after_edit(). The composite editors (stops/palette_rows)
# track their own per-sub-widget commit signal instead.
_SINGLE_ITEM_KINDS = frozenset({"enum", "bool", "int", "float", "color", "str"})

# One-char colored change-cost tag per Tier, keyed by the Tier enum's string
# value (json_schema_extra stores plain JSON, so this is a plain string).
_TIER_GLYPHS: dict[str, tuple[str, tuple[float, float, float, float], str]] = {
    "post": (
        "P",
        (0.45, 0.75, 0.45, 1.0),
        "POST -- cheap re-derive only, stays live while dragging",
    ),
    "velocity": (
        "V",
        (0.85, 0.7, 0.25, 1.0),
        "VELOCITY -- rebuilds the velocity field, sim continues",
    ),
    "restart": (
        "R",
        (0.85, 0.35, 0.35, 1.0),
        "RESTART -- re-initializes the development run from step 0",
    ),
}

_LOCK_COLOR = (0.55, 0.75, 1.0, 1.0)
_MODIFIED_COLOR = (1.0, 0.85, 0.3, 1.0)

# B4-4: exactly ONE live editor for the output settings -- the Export... modal
# (main._draw_export_modal). The auto-panel renders these read-only so two
# widgets can never again disagree on affordance or undo semantics (the old
# split: an undoable Basic slider here vs. a history-excluded snapped combo in
# the modal, which also went blank on non-preset widths).
_MODAL_ONLY_PATHS = frozenset({"export.width", "export.png_compression"})


def _draw_tier_badge(tier: Any) -> None:
    """One-char colored change-cost tag from ``extra.get('tier')`` (full word
    on hover). Sensible no-badge fallback ('.') for a leaf with no tier --
    every PlanetParams leaf is declared via ``pfield`` so none currently reach
    here without one, but the lookup stays defensive for any future
    non-pfield leaf."""
    key = tier.value if hasattr(tier, "value") else tier
    spec = _TIER_GLYPHS.get(key) if isinstance(key, str) else None
    if spec is None:
        imgui.text_disabled(".")
        return
    glyph, color, full = spec
    imgui.text_colored(imgui.ImVec4(*color), glyph)
    if imgui.is_item_hovered():
        imgui.set_tooltip(full)


def _default_value(name: str, baseline: dict[str, Any]) -> Any:
    """The default for ``name``, already dumped to the draft's JSON shape
    (``baseline`` IS that JSON shape -- a ``PlanetParams().model_dump()``).
    Deep-copied so a composite reset (palette/stops) doesn't hand back a
    reference into the cached baseline -- a later in-place mutation of the
    draft (add/remove stop, edit a color) would otherwise corrupt every future
    "Reset to default" for that field."""
    return copy.deepcopy(baseline[name])


def _draw_leaf(
    name: str,
    info: FieldInfo,
    doc: dict[str, Any],
    baseline: dict[str, Any],
    state: PanelState,
    path: str,
) -> tuple[bool, bool]:
    # Filter check BEFORE push_id, so a hidden leaf's early return stays
    # push_id/pop_id-balanced (nothing was pushed, so there's nothing to pop).
    if not _leaf_visible(name, info, doc, state):
        return False, False

    value = doc[name]
    label = name.replace("_", " ")
    meta = FieldMeta.of(info)
    lo, hi = _bounds(info)
    changed = False
    committed = False
    imgui.push_id(name)

    _draw_tier_badge(meta.tier)
    imgui.same_line()
    if path in state.locked:
        imgui.text_colored(imgui.ImVec4(*_LOCK_COLOR), "L")
        imgui.same_line()
    if name in baseline and _leaf_changed(doc[name], baseline[name]):
        imgui.text_colored(imgui.ImVec4(*_MODIFIED_COLOR), "*")
        imgui.same_line()

    if path in _MODAL_ONLY_PATHS:
        # Read-only mirror of the value; the Export... modal is the editor.
        # Early return also skips the right-click Reset (a second editor in
        # disguise) -- the tooltip still explains the field.
        imgui.text_disabled(f"{label}: {value} — set in the Export... dialog")
        if info.description and imgui.is_item_hovered():
            imgui.set_tooltip(info.description)
        imgui.pop_id()
        return False, False

    kind = leaf_kind(name, info, value)
    ann = info.annotation
    if kind == "enum":
        options = [e.value for e in ann]
        current = options.index(value) if value in options else 0
        changed, idx = imgui.combo(label, current, options)
        if changed:
            doc[name] = options[idx]
    elif kind == "bool":
        changed, doc[name] = imgui.checkbox(label, value)
    elif kind == "int":
        ilo = int(lo) if lo is not None else 0
        ihi = int(hi) if hi is not None else 100
        if ihi - ilo > 1_000_000:  # seeds etc: free input, not a slider
            changed, doc[name] = imgui.input_int(label, value)
            if changed:
                doc[name] = max(ilo, min(ihi, doc[name]))
        else:
            changed, doc[name] = imgui.slider_int(label, value, ilo, ihi)
    elif kind == "float":
        flo = lo if lo is not None else 0.0
        fhi = hi if hi is not None else 1.0
        flags = imgui.SliderFlags_.logarithmic if meta.log else 0
        changed, doc[name] = imgui.slider_float(label, value, flo, fhi, flags=flags)
    elif kind == "color":
        changed, rgb = imgui.color_edit3(label, list(value))
        if changed:
            doc[name] = tuple(rgb)
    elif kind == "str":
        changed, doc[name] = imgui.input_text(label, value)
    elif kind == "stops":
        changed, committed = _draw_stops(label, value)
    elif kind == "palette_rows":
        changed, committed = _draw_palette_rows(label, value)
    elif kind == "model_list":
        changed, committed = _draw_cast_list(label, value)
    elif kind == "optional_str":
        changed, committed = _draw_optional_str(name, label, doc)
    elif kind == "optional_float":
        changed, committed = _draw_optional_float(name, label, doc, lo, hi)
    elif kind == "optional_int":
        changed, committed = _draw_optional_int(name, label, doc, lo, hi)
    elif kind == "optional_model":
        imgui.text_disabled(f"{label}: {'set (preset-only)' if value else 'none'}")
    else:
        imgui.text_disabled(f"{label}: {value!r}")

    # For a single-item widget, "finished editing" is reported by imgui for the
    # item just drawn (slider release, Enter/focus-loss, combo/checkbox/color
    # pick). Composite kinds set `committed` themselves above.
    if kind in _SINGLE_ITEM_KINDS:
        committed = imgui.is_item_deactivated_after_edit()

    if info.description and imgui.is_item_hovered():
        imgui.set_tooltip(info.description)

    # Right-click affordances, tied to the leaf's last-drawn item.
    #
    # Explicit str_id ("context"): for int leaves rendered via input_int
    # (seeds etc.), the +/- stepper wraps the widget in BeginGroup/EndGroup,
    # and EndGroup's closing ItemAdd registers id=0 as the last item --
    # begin_popup_context_item() with no str_id falls back to that last-item
    # id and hits imgui's IM_ASSERT(id != 0) every frame. The literal
    # "context" here is fine (not the field's dotted path) because we're
    # already inside this leaf's push_id(name) scope (see above), so the
    # resulting popup id is unique per field via the id stack.
    if imgui.begin_popup_context_item("context"):
        if imgui.menu_item_simple("Reset to default") and name in baseline:
            doc[name] = _default_value(name, baseline)
            changed = True
            committed = True
        clicked, now_locked = imgui.menu_item("Lock for randomize", "", path in state.locked)
        if clicked:
            if now_locked:
                state.locked.add(path)
            else:
                state.locked.discard(path)
        imgui.end_popup()

    imgui.pop_id()
    return changed, committed


def _draw_optional_str(
    name: str, label: str, doc: dict[str, Any]
) -> tuple[bool, bool]:
    """Text entry + ``Browse...`` button for an optional string path
    (``mask.file``). Editing writes the (absolute) path into the draft; an empty
    string clears it to None (no mask). The Browse button opens the native file
    picker and commits the chosen ABSOLUTE path. Returns ``(changed, committed)``
    like the other composite editors."""
    from imgui_bundle import portable_file_dialogs as pfd

    changed = committed = False
    current = doc[name] or ""
    c, text = imgui.input_text(label, current)
    if c:
        doc[name] = text or None
        changed = True
    committed = committed or imgui.is_item_deactivated_after_edit()
    imgui.same_line()
    if imgui.button(f"Browse...##{name}"):
        picked = pfd.open_file(
            "Select mask image", "",
            ["PNG images", "*.png", "All files", "*"],
        ).result()
        if picked:
            doc[name] = str(Path(picked[0]).resolve())
            changed = committed = True
    return changed, committed


def _optional_float_bounds(
    name: str, doc: dict[str, Any], lo: float | None, hi: float | None
) -> tuple[float, float]:
    """Slider bounds for an optional-float leaf. For ``storms.hero_latitude``
    the field bounds are tightened by the same radius-coupled cap the model
    validator enforces (``hero_latitude_cap``), read live from the sibling
    ``hero_radius`` in the draft -- so the widget can never offer a value the
    commit would reject with a validation toast (B4-2)."""
    flo = lo if lo is not None else 0.0
    fhi = hi if hi is not None else 1.0
    if name == "hero_latitude" and "hero_radius" in doc:
        cap = hero_latitude_cap(float(doc["hero_radius"]))
        flo, fhi = max(flo, -cap), min(fhi, cap)
    return flo, fhi


def _draw_optional_float(
    name: str, label: str, doc: dict[str, Any], lo: float | None, hi: float | None
) -> tuple[bool, bool]:
    """B4-2: a real widget for an optional float (``storms.hero_latitude``),
    replacing the old read-only text. A "pin" checkbox toggles None (seeded/
    auto placement) <-> a pinned value; while pinned, a slider edits the value
    inside validator-safe bounds. Pinning starts at 0.0 clamped into bounds
    (always valid: the cap is symmetric about 0 and strictly positive for
    every legal hero_radius). Returns ``(changed, committed)`` like the other
    composite editors."""
    changed = committed = False
    flo, fhi = _optional_float_bounds(name, doc, lo, hi)
    pinned = doc[name] is not None
    clicked, want_pin = imgui.checkbox(f"pin##{name}", pinned)
    if clicked and want_pin != pinned:
        doc[name] = min(max(0.0, flo), fhi) if want_pin else None
        changed = committed = True
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "pinned: the slider value is used verbatim; unpinned: seeded auto placement"
        )
    imgui.same_line()
    if doc[name] is not None:
        c, v = imgui.slider_float(label, float(doc[name]), flo, fhi)
        if c:
            doc[name] = v
            changed = True
        committed = committed or imgui.is_item_deactivated_after_edit()
    else:
        imgui.text_disabled(f"{label}: none (auto)")
    return changed, committed


def _optional_int_bounds(
    name: str, doc: dict[str, Any], lo: float | None, hi: float | None
) -> tuple[int, int]:
    """Slider bounds for an optional-int leaf. For ``bands.faded_band_index``
    the ceiling is the live band count from the sibling draft fields
    (template length when a template is set, else ``count``) -- the same rule
    the model validator enforces -- so the widget can never offer an index
    the commit would reject with a validation toast (B4-2 discipline)."""
    ilo = int(lo) if lo is not None else 0
    ihi = int(hi) if hi is not None else 100
    if name == "faded_band_index":
        template = doc.get("template")
        if isinstance(template, dict) and template.get("values"):
            ihi = min(ihi, len(template["values"]) - 1)
        elif "count" in doc:
            ihi = min(ihi, int(doc["count"]) - 1)
    return ilo, ihi


def _draw_optional_int(
    name: str, label: str, doc: dict[str, Any], lo: float | None, hi: float | None
) -> tuple[bool, bool]:
    """Optional-int twin of ``_draw_optional_float`` (``bands.faded_band_index``):
    a "pin" checkbox toggles None (auto selection) <-> an explicit value edited
    with a validator-safe int slider. Pinning starts at the low bound."""
    changed = committed = False
    ilo, ihi = _optional_int_bounds(name, doc, lo, hi)
    pinned = doc[name] is not None
    clicked, want_pin = imgui.checkbox(f"pin##{name}", pinned)
    if clicked and want_pin != pinned:
        doc[name] = ilo if want_pin else None
        changed = committed = True
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "pinned: the slider value is used verbatim; unpinned: automatic selection"
        )
    imgui.same_line()
    if doc[name] is not None:
        c, v = imgui.slider_int(label, int(doc[name]), ilo, ihi)
        if c:
            doc[name] = v
            changed = True
        committed = committed or imgui.is_item_deactivated_after_edit()
    else:
        imgui.text_disabled(f"{label}: none (auto)")
    return changed, committed


def _draw_palette_rows(label: str, rows: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Latitude-anchored palette rows: a latitude slider plus the shared
    stops editor per row.

    Returns ``(changed, committed)``. ``committed`` is the OR of every
    sub-widget's end-of-edit signal plus a synthetic commit on any structural
    mutation (add/remove row) — plain buttons never raise the imgui signal, so
    without this an add/remove would dangle the caller's gesture base.
    """
    changed = False
    committed = False
    imgui.text(label)
    remove_index = None
    for i, row in enumerate(rows):
        imgui.push_id(1000 + i)
        imgui.separator_text(f"row {i + 1}")
        imgui.set_next_item_width(160.0)
        c, lat = imgui.slider_float("latitude", float(row["latitude"]), -90.0, 90.0)
        if c:
            row["latitude"] = lat
            changed = True
        committed |= imgui.is_item_deactivated_after_edit()
        if imgui.is_item_hovered():
            # B2-3: the composite editors' sub-widgets have no pfield
            # description to inherit, so they carry their own tooltips.
            imgui.set_tooltip(
                "Anchor latitude of this gradient row, degrees (north positive); "
                "rows blend into each other across latitude"
            )
        if len(rows) > 1:
            imgui.same_line()
            if imgui.small_button("remove row"):
                remove_index = i
        sc, scommitted = _draw_stops("stops", row["stops"])
        changed |= sc
        committed |= scommitted
        imgui.pop_id()
    if remove_index is not None:
        rows.pop(remove_index)
        changed = True
        committed = True
    if imgui.small_button(f"add row##{label}"):
        last = rows[-1]
        rows.append(
            {
                "latitude": min(90.0, float(last["latitude"]) + 30.0),
                # copy each stop's color too (dict(s) is shallow) so the new row
                # never aliases the source row's color objects
                "stops": [{**s, "color": tuple(s["color"])} for s in last["stops"]],
            }
        )
        changed = True
        committed = True
    return changed, committed


def _draw_cast_list(label: str, rows: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Editor for the art-directed cast list (``storms.cast``): one row per
    StormOverride with a kind combo, position/size/strength/aspect sliders, and
    optional tint/brightness overrides (a checkbox toggles None = kind default
    vs an explicit value). Add/remove rows; the list may be empty. Returns
    ``(changed, committed)`` with the same semantics as ``_draw_palette_rows``
    (per-sub-widget release OR a synthetic commit on a structural mutation --
    plain buttons never raise the imgui end-of-edit signal)."""
    from gasgiant.params.model import CastKind, StormOverride

    changed = False
    committed = False
    kinds = [k.value for k in CastKind]
    imgui.text(label)
    remove_index = None
    for i, row in enumerate(rows):
        imgui.push_id(2000 + i)
        imgui.separator_text(f"storm {i + 1}")
        cur = kinds.index(row["kind"]) if row["kind"] in kinds else 0
        c, idx = imgui.combo("kind", cur, kinds)
        if c:
            row["kind"] = kinds[idx]
            changed = True
        committed |= imgui.is_item_deactivated_after_edit()
        for key, lo, hi, lbl in (
            ("lat_deg", -68.0, 68.0, "latitude"),
            ("lon_deg", -180.0, 180.0, "longitude"),
            ("radius", 0.01, 0.15, "radius"),
            ("strength_scale", 0.0, 3.0, "strength"),
            ("aspect", 1.0, 3.0, "aspect"),
        ):
            cc, v = imgui.slider_float(lbl, float(row[key]), lo, hi)
            if cc:
                row[key] = v
                changed = True
            committed |= imgui.is_item_deactivated_after_edit()
        for key, lo, hi in (("tint", -1.0, 1.0), ("brightness", -0.5, 0.5)):
            enabled = row[key] is not None
            ec, want = imgui.checkbox(f"set {key}##{key}", enabled)
            if ec and want != enabled:
                row[key] = 0.0 if want else None
                changed = committed = True
            imgui.same_line()
            if row[key] is not None:
                sc, sv = imgui.slider_float(key, float(row[key]), lo, hi)
                if sc:
                    row[key] = sv
                    changed = True
                committed |= imgui.is_item_deactivated_after_edit()
            else:
                imgui.text_disabled(f"{key}: kind default")
        if imgui.small_button("remove storm"):
            remove_index = i
        imgui.pop_id()
    if remove_index is not None:
        rows.pop(remove_index)
        changed = True
        committed = True
    if imgui.small_button(f"add storm##{label}"):
        rows.append(StormOverride().model_dump())
        changed = True
        committed = True
    return changed, committed


def _draw_stops(label: str, stops: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Minimal gradient-stop editor (the full palette editor lands in Phase 3c).

    Returns ``(changed, committed)`` — see ``_draw_palette_rows`` for the commit
    semantics (per-sub-widget release OR a synthetic commit on add/remove stop).
    """
    changed = False
    committed = False
    imgui.text(label)
    remove_index = None
    for i, stop in enumerate(stops):
        imgui.push_id(i)
        imgui.set_next_item_width(140.0)
        c, rgb = imgui.color_edit3("##c", list(stop["color"]))
        if c:
            stop["color"] = tuple(rgb)
            changed = True
        committed |= imgui.is_item_deactivated_after_edit()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Color of this gradient stop (sRGB)")
        imgui.same_line()
        imgui.set_next_item_width(120.0)
        c, pos = imgui.slider_float("##p", float(stop["pos"]), 0.0, 1.0)
        if c:
            stop["pos"] = pos
            changed = True
        committed |= imgui.is_item_deactivated_after_edit()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Position of this stop along the gradient: 0 = darkest belt "
                "end, 1 = brightest zone end (for storm_tints: 0 = festoon "
                "blue-gray end, 1 = reddest storm end)"
            )
        if len(stops) > 1:
            imgui.same_line()
            if imgui.small_button("x"):
                remove_index = i
        imgui.pop_id()
    if remove_index is not None:
        stops.pop(remove_index)
        changed = True
        committed = True
    if imgui.small_button(f"add stop##{label}"):
        last = stops[-1]
        # copy the color so the new stop never aliases the source stop's object
        stops.append({"pos": min(1.0, last["pos"] + 0.1), "color": tuple(last["color"])})
        changed = True
        committed = True
    return changed, committed

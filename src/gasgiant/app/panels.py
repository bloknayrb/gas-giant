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
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import annotated_types
from imgui_bundle import imgui
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from gasgiant.params.model import PlanetParams


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
    - ``show_advanced`` -- plumbed now so Phase 4 doesn't need to touch this
      dataclass again; THIS phase does not gate any visibility on it.
    - ``locked`` -- dotted field paths excluded from ``randomize()`` (the
      panel's right-click "Lock for randomize" toggle and the header seed
      lock both write here).
    """

    search: str = ""
    show_advanced: bool = True
    locked: set[str] = dataclasses.field(default_factory=set)


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
    "emission": "Self-emissive glow channels (lightning, aurora, hot spots).",
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
    imgui.set_next_item_width(-30.0)
    changed, text = imgui.input_text_with_hint("##panel_search", "search fields...", state.search)
    if changed:
        state.search = text
    imgui.same_line()
    if imgui.button("X##clear_search"):
        state.search = ""
    imgui.separator()


def _leaf_visible(name: str, info: FieldInfo, doc: dict[str, Any], state: PanelState) -> bool:
    """Search-filter predicate. Shared verbatim by the leaf draw (skip
    drawing a non-matching leaf) AND the section pre-pass (skip a section
    with zero matching leaves) -- the SAME function, not two copies that could
    drift. Matches the field name, its display label, and its description,
    case-insensitive substring. ``doc`` isn't used by the match itself (kept
    in the signature for symmetry with the draw call site / future filters).
    """
    query = state.search.strip().lower()
    if not query:
        return True
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
    for name, info in model.model_fields.items():
        ann = info.annotation
        path = f"{prefix}{name}"
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            # Zero-match section suppression is search-only: outside a search
            # every section header still renders (a fully-advanced section
            # like Solver must show its header in plain browsing).
            if searching and not _subtree_has_match(ann, doc[name], state):
                continue
            flags = imgui.TreeNodeFlags_.default_open if top_level else 0
            opened = _section_header(name.capitalize(), flags, searching)
            if top_level:
                blurb = _SECTION_BLURBS.get(name)
                if blurb:
                    imgui.same_line()
                    _draw_help_marker(blurb)
            if opened:
                imgui.push_id(name)
                imgui.indent(8.0)
                c, cm = _draw_model(ann, doc[name], baseline[name], state, prefix=f"{path}.")
                changed |= c
                committed |= cm
                imgui.unindent(8.0)
                imgui.pop_id()
            continue
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
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
    lo, hi = _bounds(info)
    changed = False
    committed = False
    imgui.push_id(name)

    _draw_tier_badge(extra.get("tier"))
    imgui.same_line()
    if path in state.locked:
        imgui.text_colored(imgui.ImVec4(*_LOCK_COLOR), "L")
        imgui.same_line()
    if name in baseline and doc[name] != baseline[name]:
        imgui.text_colored(imgui.ImVec4(*_MODIFIED_COLOR), "*")
        imgui.same_line()

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
        flags = imgui.SliderFlags_.logarithmic if extra.get("log") else 0
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
    elif kind == "optional_float":
        imgui.text_disabled(f"{label}: {value if value is not None else 'none (auto)'}")
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
    if imgui.begin_popup_context_item():
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
                "stops": [dict(s) for s in last["stops"]],
            }
        )
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
        imgui.same_line()
        imgui.set_next_item_width(120.0)
        c, pos = imgui.slider_float("##p", float(stop["pos"]), 0.0, 1.0)
        if c:
            stop["pos"] = pos
            changed = True
        committed |= imgui.is_item_deactivated_after_edit()
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
        stops.append({"pos": min(1.0, last["pos"] + 0.1), "color": last["color"]})
        changed = True
        committed = True
    return changed, committed

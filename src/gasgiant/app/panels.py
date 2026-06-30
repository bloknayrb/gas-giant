"""Auto-generated parameter panels.

Widgets are derived from the pydantic model: nested models become collapsing
headers, numeric fields become sliders bounded by their validation constraints,
color tuples become color pickers, gradient-stop lists get a simple stops
editor. Adding a sim parameter in params/model.py is all it takes to get UI.

Edits mutate a plain dict draft; the caller validates and commits. The engine
never sees a mid-drag invalid state.
"""

from __future__ import annotations

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


def draw_params_panel(params: PlanetParams) -> tuple[dict[str, Any], bool, bool]:
    """Draw all parameter widgets against a fresh draft of ``params``.

    Returns ``(draft, any_changed, any_committed)``:

    - ``any_changed``  — a value changed this frame (mid-drag included).
    - ``any_committed`` — a widget *finished* editing this frame (slider
      release, combo/checkbox/color pick, ``input_int`` Enter/focus-loss, or a
      structural palette mutation). The caller defers heavy (velocity/restart)
      rebuilds to ``any_committed`` frames so a drag commits once on release
      instead of every frame.
    """
    draft = params.model_dump()
    changed, committed = _draw_model(type(params), draft, top_level=True)
    return draft, changed, committed


def _draw_model(
    model: type[BaseModel], doc: dict[str, Any], top_level: bool = False
) -> tuple[bool, bool]:
    changed = False
    committed = False
    for name, info in model.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            flags = imgui.TreeNodeFlags_.default_open if top_level else 0
            if imgui.collapsing_header(name.capitalize(), flags):
                imgui.push_id(name)
                imgui.indent(8.0)
                c, cm = _draw_model(ann, doc[name])
                changed |= c
                committed |= cm
                imgui.unindent(8.0)
                imgui.pop_id()
            continue
        c, cm = _draw_leaf(name, info, doc)
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


def _draw_leaf(name: str, info: FieldInfo, doc: dict[str, Any]) -> tuple[bool, bool]:
    value = doc[name]
    label = name.replace("_", " ")
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
    lo, hi = _bounds(info)
    changed = False
    committed = False
    imgui.push_id(name)

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

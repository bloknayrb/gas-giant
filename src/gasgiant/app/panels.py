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


def draw_params_panel(params: PlanetParams) -> dict[str, Any] | None:
    """Draw all parameter widgets. Returns the edited draft dict if anything
    changed this frame, else None."""
    draft = params.model_dump()
    changed = _draw_model(type(params), draft, top_level=True)
    return draft if changed else None


def _draw_model(model: type[BaseModel], doc: dict[str, Any], top_level: bool = False) -> bool:
    changed = False
    for name, info in model.model_fields.items():
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            flags = imgui.TreeNodeFlags_.default_open if top_level else 0
            if imgui.collapsing_header(name.capitalize(), flags):
                imgui.push_id(name)
                imgui.indent(8.0)
                changed |= _draw_model(ann, doc[name])
                imgui.unindent(8.0)
                imgui.pop_id()
            continue
        changed |= _draw_leaf(name, info, doc)
    return changed


def leaf_kind(name: str, info: FieldInfo, value: Any) -> str | None:
    """Which widget _draw_leaf renders for this field, or None if it has no
    widget. Kept as a pure function so a static test can assert every leaf
    in PlanetParams is renderable without opening a GUI."""
    from enum import StrEnum

    ann = info.annotation
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


def _draw_leaf(name: str, info: FieldInfo, doc: dict[str, Any]) -> bool:
    value = doc[name]
    label = name.replace("_", " ")
    extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
    lo, hi = _bounds(info)
    changed = False
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
        changed = _draw_stops(label, value)
    elif kind == "palette_rows":
        changed = _draw_palette_rows(label, value)
    else:
        imgui.text_disabled(f"{label}: {value!r}")

    if info.description and imgui.is_item_hovered():
        imgui.set_tooltip(info.description)
    imgui.pop_id()
    return changed


def _draw_palette_rows(label: str, rows: list[dict[str, Any]]) -> bool:
    """Latitude-anchored palette rows: a latitude slider plus the shared
    stops editor per row."""
    changed = False
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
        if len(rows) > 1:
            imgui.same_line()
            if imgui.small_button("remove row"):
                remove_index = i
        changed |= _draw_stops("stops", row["stops"])
        imgui.pop_id()
    if remove_index is not None:
        rows.pop(remove_index)
        changed = True
    if imgui.small_button(f"add row##{label}"):
        last = rows[-1]
        rows.append(
            {
                "latitude": min(90.0, float(last["latitude"]) + 30.0),
                "stops": [dict(s) for s in last["stops"]],
            }
        )
        changed = True
    return changed


def _draw_stops(label: str, stops: list[dict[str, Any]]) -> bool:
    """Minimal gradient-stop editor (the full palette editor lands in Phase 3c)."""
    changed = False
    imgui.text(label)
    remove_index = None
    for i, stop in enumerate(stops):
        imgui.push_id(i)
        imgui.set_next_item_width(140.0)
        c, rgb = imgui.color_edit3("##c", list(stop["color"]))
        if c:
            stop["color"] = tuple(rgb)
            changed = True
        imgui.same_line()
        imgui.set_next_item_width(120.0)
        c, pos = imgui.slider_float("##p", float(stop["pos"]), 0.0, 1.0)
        if c:
            stop["pos"] = pos
            changed = True
        if len(stops) > 1:
            imgui.same_line()
            if imgui.small_button("x"):
                remove_index = i
        imgui.pop_id()
    if remove_index is not None:
        stops.pop(remove_index)
        changed = True
    if imgui.small_button(f"add stop##{label}"):
        last = stops[-1]
        stops.append({"pos": min(1.0, last["pos"] + 0.1), "color": last["color"]})
        changed = True
    return changed

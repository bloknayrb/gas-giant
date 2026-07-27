"""Architectural guard: ``gasgiant.app.tooltips`` is the only module allowed to
open a tooltip window.

Deliberately dependency-free -- no imgui, no ``importorskip``. Its sibling
``test_tooltip_wrapping.py`` skips wholesale without the GUI extra, and an
architectural guard that disappears on a contributor's box is exactly the
silent gap it exists to close.

Belt-and-braces with ruff's ``TID251`` ban (``pyproject.toml``), which is the
primary mechanism -- it scans subpackages this test's top-level glob misses and
fires in-editor. The two have complementary blind spots: ruff cannot resolve an
aliased module (``im = imgui; im.set_tooltip(...)``), and this scan cannot see
``getattr(imgui, "set_tooltip")``.
"""

from __future__ import annotations

import ast
import pathlib

# Opening a tooltip window without pushing the wrap is the regression here, so
# the begin_* forms are banned alongside set_tooltip -- a set_tooltip-only
# check would be blind to a bare begin_tooltip().
BANNED = {
    "set_tooltip",
    "set_item_tooltip",
    "begin_tooltip",
    "begin_item_tooltip",
    "begin_tooltip_ex",
}


def _called_name(node: ast.Call) -> str | None:
    """The bare callable name, from either ``imgui.set_tooltip(...)`` or a
    ``from imgui_bundle.imgui import set_tooltip`` style direct call."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_no_raw_tooltip_calls_survive_in_the_app_layer():
    import gasgiant.app

    app_dir = pathlib.Path(gasgiant.app.__file__).parent
    offenders: list[str] = []
    for path in sorted(app_dir.rglob("*.py")):
        if path.name == "tooltips.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders += [
            f"{path.name}:{node.lineno} {_called_name(node)}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and _called_name(node) in BANNED
        ]
    assert not offenders, f"use tooltips.item_tooltip/tooltip instead: {offenders}"

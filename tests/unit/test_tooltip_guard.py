"""Architectural guard: ``gasgiant.app.tooltips`` is the only module allowed to
open a tooltip window.

Deliberately dependency-free -- no imgui, no ``importorskip``. Its sibling
``test_tooltip_wrapping.py`` skips wholesale without the GUI extra, and an
architectural guard that disappears on a contributor's box is exactly the
silent gap it exists to close.

Belt-and-braces with ruff's ``TID251`` ban (``pyproject.toml``), which is the
primary mechanism -- it fires in-editor and on every ``ruff check``. The two
are genuinely complementary, though not in the obvious way: ruff DOES resolve
an import alias (``from imgui_bundle import imgui as im``), and only a runtime
rebinding (``im = imgui`` inside a function) evades it, which this name-based
scan still catches. Neither sees ``getattr(imgui, "set_tooltip")``.
"""

from __future__ import annotations

import ast
import pathlib
import tomllib

_PYPROJECT = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"


def _banned_names() -> set[str]:
    """The banned callables, read from ruff's own ``TID251`` table.

    Derived rather than restated: a hardcoded copy drifts the moment a sixth
    entry point is added to ``pyproject.toml``, and it drifts SILENTLY -- this
    guard exists to catch what ruff cannot see, so a stale list defeats its
    whole purpose with nothing red to say so. ``tomllib`` is stdlib on 3.13, so
    reading it keeps this module dependency-free.

    Opening a tooltip window without pushing the wrap is the regression here,
    so the table bans the ``begin_*`` forms alongside ``set_tooltip`` -- a
    ``set_tooltip``-only check would be blind to a bare ``begin_tooltip()``.
    """
    cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    table = cfg["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
    return {key.rsplit(".", 1)[-1] for key in table}


BANNED = _banned_names()


def test_the_banned_list_is_populated():
    """Without this, a renamed or deleted ruff table would empty ``BANNED`` and
    turn the scan below into a test that passes by checking nothing."""
    assert {"set_tooltip", "begin_tooltip"} <= BANNED, BANNED


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

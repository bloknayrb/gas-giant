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

    Derived so a SIXTH entry added to ``pyproject.toml`` flows in for free.
    Removals are covered the other way, by ``REQUIRED`` below -- derivation
    alone would let a ruff-table deletion disarm both mechanisms at once.
    ``tomllib`` is stdlib on 3.13, so reading it keeps this module
    dependency-free.

    A ``KeyError`` here is deliberate: if the table is renamed or removed the
    module fails loudly at import rather than yielding an empty set and a scan
    that passes by checking nothing.

    Opening a tooltip window without pushing the wrap is the regression here,
    so the table bans the ``begin_*`` forms alongside ``set_tooltip`` -- a
    ``set_tooltip``-only check would be blind to a bare ``begin_tooltip()``.
    """
    cfg = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    table = cfg["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-api"]
    return {key.rsplit(".", 1)[-1] for key in table}


BANNED = _banned_names()


#: The tooltip openers this module exists to keep out of the app layer, pinned
#: independently of the ruff table.
#:
#: Deriving BANNED without this pin QUIETLY UNDID the belt-and-braces: with both
#: mechanisms reading one list, deleting a line from ruff's table disarms them
#: TOGETHER. Verified end to end -- drop the ``begin_item_tooltip`` entry from
#: pyproject and a raw ``imgui.begin_item_tooltip()`` in panels.py passes ruff,
#: passes this scan, and passes the whole fast tier. Before the derivation the
#: hardcoded set kept the AST scan red in exactly that case, which is what
#: "complementary" meant. Derivation still earns its place -- a NEW entry flows
#: in automatically -- but a removal must not go unnoticed.
REQUIRED = {
    "set_tooltip",
    "set_item_tooltip",
    "begin_tooltip",
    "begin_item_tooltip",
    "begin_tooltip_ex",
}


def test_the_ruff_table_still_bans_every_known_opener():
    assert REQUIRED <= BANNED, f"ruff's banned-api table lost {sorted(REQUIRED - BANNED)}"


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
            if isinstance(node, ast.Call) and _called_name(node) in (BANNED | REQUIRED)
        ]
    assert not offenders, f"use tooltips.item_tooltip/tooltip instead: {offenders}"

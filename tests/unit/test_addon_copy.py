"""The Blender importer's artist-facing copy, checked without Blender.

``blender_addon/`` is the one artist-facing surface with no test reach at all:
``tests/blender/test_import.py`` only runs under ``blender --background``, which
is neither on PATH nor in CI, and it asserts nothing about copy in any case. So
the importer's operator panel — every string a Blender user reads — could go
empty or turn to jargon with nothing red to say so.

This scrape is deliberately AST-only. Importing the module needs ``bpy``, which
exists solely inside Blender, so the alternative to a scrape is no check.

Blender renders a property's ``description`` as the hover tooltip and wraps it
itself, so the wrapping work in ``gasgiant.app.tooltips`` does not transfer here
and length is bounded only against runaway paragraphs. House convention in this
file is no trailing period, which is Blender's own.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_IMPORTER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "blender_addon" / "gasgiant_importer" / "importer.py"
)

#: Blender property constructors whose ``description=`` is the user-visible
#: tooltip. Listed rather than matched loosely so a NEW property type has to be
#: added here consciously.
_PROPERTY_CALLS = {
    "FloatProperty", "IntProperty", "BoolProperty", "StringProperty", "EnumProperty",
}

#: ``filter_glob`` is a hidden file-dialog filter, never drawn, so it carries no
#: description by design.
_NOT_USER_FACING = {"filter_glob"}

_MAX_DESCRIPTION = 400


#: Parsed once, at import; every check below walks this same tree. A ``_tree()``
#: helper re-read and re-parsed importer.py on each of its four call sites, and
#: nothing here mutates the tree, so the four copies were identical by
#: construction.
_TREE = ast.parse(_IMPORTER.read_text(encoding="utf-8"))


def _properties() -> list[tuple[str, str | None]]:
    """``(property name, description or None)`` for every declared property."""
    out: list[tuple[str, str | None]] = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        call = node.annotation  # `name: FloatProperty(...)` -- the CALL is the annotation
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in _PROPERTY_CALLS:
            continue
        desc = None
        for kw in call.keywords:
            if (kw.arg == "description" and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)):
                desc = kw.value.value
        out.append((node.target.id, desc))
    return out


#: Scraped once and shared: the parametrize below needs it at collection time,
#: and the scrape guard must assert against the SAME list the cases run on.
_PROPERTIES = _properties()


def test_the_scrape_actually_finds_the_operator_properties():
    """Without this the whole module passes by checking an empty list -- the
    exact failure mode a scrape-based guard is prone to."""
    names = [name for name, _ in _PROPERTIES]
    assert len(names) >= 12, f"scrape found only {names}"
    assert {"radius", "mesh_segments", "build_rings", "axial_tilt"} <= set(names)


@pytest.mark.parametrize(
    ("name", "desc"),
    [(n, d) for n, d in _PROPERTIES if n not in _NOT_USER_FACING],
    ids=[n for n, _ in _PROPERTIES if n not in _NOT_USER_FACING],
)
def test_every_user_facing_property_is_described(name, desc):
    """FIVE properties shipped with no description at all -- limb_darkening,
    limb_haze, axial_tilt, and both enums (mapping, atmosphere_mode, whose
    ITEMS had tooltips while the property itself did not). All drawn in the
    import panel with an empty tooltip."""
    assert desc, f"{name} is drawn in the import panel with no tooltip"
    assert len(desc) <= _MAX_DESCRIPTION, f"{name}: {len(desc)} chars is a paragraph"
    assert not desc.endswith("."), f"{name}: Blender convention is no trailing period"


def test_the_operator_itself_is_described():
    """``bl_description`` is the File > Import menu tooltip -- the single most
    visible string in the addon, and the one a ``description=`` grep misses."""
    for node in ast.walk(_TREE):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "bl_description" for t in node.targets
        ):
            assert isinstance(node.value, ast.Constant) and node.value.value.strip()
            assert not node.value.value.endswith(".")
            return
    pytest.fail("no bl_description on the import operator")


def test_enum_items_carry_their_own_descriptions():
    """A Blender enum item is ``(id, label, description)``; an empty third slot
    is a dropdown entry with no tooltip. 'None' options legitimately have
    nothing to say, so only non-trivial options are required to speak."""
    empty: list[str] = []
    checked: list[str] = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Tuple) or len(node.elts) != 3:
            continue
        if not all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            continue
        ident, label, desc = (e.value for e in node.elts)
        if not ident.isupper():
            continue
        checked.append(ident)
        if ident != "NONE" and not desc.strip():
            empty.append(f"{ident} ({label})")
    # Without this the selector is a silent skip: lowercasing an option id
    # drops it from the scan, tooltip or no tooltip, and the test still passes.
    assert len(checked) >= 4, f"enum scrape found only {checked}"
    assert not empty, f"enum options with no tooltip: {empty}"

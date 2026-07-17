"""GPU-free projection guards for the render kernels' preprocessor variants.

Two guards, both built on a no-GL projection of the flattened source:

1. **Default-program drift** (``test_default_program_projection_is_stable``).
   The SPREAD variant wraps several base-path lines in ``#ifdef SPREAD /
   #else / #endif`` with the ``#else`` arm a VERBATIM copy of today's line. The GPU
   routing byte-identity test proves the default OUTPUT is unchanged *now*; this
   unit test locks it cheaply in CI (no GL context) going forward by evaluating the
   NO-DEFINES preprocessor projection of the flattened source and hashing it. A
   non-verbatim ``#else`` arm (or an accidental base-path edit) moves the hash.

2. **Variant declare/use mismatch** (``test_every_variant_declares_the_uniforms_it_uses``).
   A uniform must be declared under the LOOSEST guard any of its uses sit under.
   Violating that only breaks the variants where the declaration is stripped but a
   use survives -- which compiles fine in every combination the tests happen to
   exercise and crashes for the user who reaches the odd one. This is not
   hypothetical: ``u_hero_emergence`` was declared inside ``#ifdef DETAIL_FX`` while
   its quiet-storm uses sat in the base path under ``#ifdef HERO_EMERGENCE`` alone,
   so the (DETAIL_FX off, HERO_EMERGENCE on) variant -- a default planet with the
   emergence slider raised -- died with ``undefined variable "u_hero_emergence"``.
   Because the defect is purely TEXTUAL, a projection catches it with no GL context,
   in the always-blocking no-GPU tier, across the whole reachable variant matrix.
   The GPU-side compile test (tests/gpu/test_detail_fx.py) proves real compilation;
   this proves the invariant on every PR.

``gl/context._load_flattened`` only expands ``#include``; it leaves the
``#ifdef`` directives as literal text (the GLSL compiler resolves them). So we
apply a minimal C-style preprocessor here for the exact directive subset the
kernels use (``#ifdef``/``#ifndef``/``#if defined(...)``/``#else``/``#endif``)."""
from __future__ import annotations

import hashlib
import itertools
import re

import pytest

from gasgiant.gl.context import _load_flattened


def _eval_if(expr: str, defines: set[str]) -> bool:
    expr = expr.strip()
    # Flat boolean combinations of defined() terms (no parenthesized grouping
    # in any kernel): || binds looser than &&, so split on it first.
    if "||" in expr:
        return any(_eval_if(t, defines) for t in expr.split("||"))
    if "&&" in expr:
        return all(_eval_if(t, defines) for t in expr.split("&&"))
    if expr.startswith("defined(") and expr.endswith(")"):
        return expr[len("defined("):-1].strip() in defines
    if expr.startswith("defined "):
        return expr[len("defined "):].strip() in defines
    raise ValueError(f"unsupported #if expression: {expr!r}")


def _preprocess(source: str, defines: set[str]) -> str:
    """Project the flattened source through the given defines. Handles the
    directive subset the kernels use, with nesting. ``#define`` lines are passed
    through as text (not evaluated) -- fine for a deterministic projection."""
    out: list[str] = []
    stack: list[dict] = []  # frames: {parent, taken, active}

    def emitting() -> bool:
        return all(f["active"] for f in stack)

    for line in source.splitlines():
        s = line.strip()
        if s.startswith("#ifdef "):
            parent = emitting()
            active = parent and (s[len("#ifdef "):].strip() in defines)
            stack.append({"parent": parent, "taken": active, "active": active})
        elif s.startswith("#ifndef "):
            parent = emitting()
            active = parent and (s[len("#ifndef "):].strip() not in defines)
            stack.append({"parent": parent, "taken": active, "active": active})
        elif s.startswith("#if "):
            parent = emitting()
            active = parent and _eval_if(s[len("#if "):], defines)
            stack.append({"parent": parent, "taken": active, "active": active})
        elif s == "#else":
            f = stack[-1]
            f["active"] = f["parent"] and not f["taken"]
            f["taken"] = f["taken"] or f["active"]
        elif s == "#endif":
            stack.pop()
        elif emitting():
            out.append(line)
    return "\n".join(out) + "\n"


def _no_defines_projection() -> str:
    source, _ = _load_flattened("gasgiant.render.kernels", "detail.comp", {})
    return _preprocess(source, set())


# Hash of the NO-DEFINES projection. Regenerate DELIBERATELY (paste the printed
# value) ONLY when a real base-path change is intended.
GOLDEN = "6e5b8dea2b676ca14fcb17a2b970472f3e7513db7896cde79f0402f095686172"


def test_default_program_projection_is_stable():
    digest = hashlib.sha256(_no_defines_projection().encode("utf-8")).hexdigest()
    assert digest == GOLDEN, digest


def test_variant_only_symbols_absent_from_default_projection():
    """Semantic check: the default projection must strip BOTH the SPREAD block
    (u_spread) and the DETAIL_FX block (u_belt_texture/u_mottle)."""
    proj = _no_defines_projection()
    for sym in ("u_spread", "u_belt_texture", "u_mottle"):
        assert sym not in proj, f"{sym} leaked into the default (no-defines) program"


# ------------------------------------------- variant declare/use matrix guard

# Render kernels whose variant axes are ALL boolean (``#ifdef``/``#if defined``),
# i.e. the subset ``_preprocess`` evaluates. The sim kernels are deliberately out
# of scope: they gate on VALUE defines (``#if DOMAIN == 0``, ``#elif SUBPASS == 1``)
# that need a real expression evaluator, and _eval_if raises on those rather than
# guessing. Their variants are covered by the sim tests + the GPU compile path.
# test_variant_axes_are_complete keeps these lists honest.
_VARIANT_AXES: dict[tuple[str, str], tuple[str, ...]] = {
    ("gasgiant.render.kernels", "detail.comp"): ("DETAIL_FX", "SPREAD", "HERO_EMERGENCE"),
    ("gasgiant.render.kernels", "derive.comp"): (
        "EMISSION", "CHROMA_FX", "MASK", "BAND_TINT", "PROJECTION_CUBE", "DETAIL_CHROMA",
    ),
}

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_LINE_COMMENT = re.compile(r"//[^\n]*")
# ``uniform sampler2D u_x;`` / ``uniform float u_spin[3];`` / layout-qualified forms.
_UNIFORM_DECL = re.compile(r"\buniform\s+\w+\s+(u_\w+)")
_UNIFORM_USE = re.compile(r"\bu_\w+\b")
_GUARD_IFDEF = re.compile(r"^\s*#(?:ifdef|ifndef)\s+(\w+)", re.M)
_GUARD_DEFINED = re.compile(r"\bdefined\s*\(\s*(\w+)\s*\)")


def _strip_comments(src: str) -> str:
    """Comments name uniforms prolifically in these kernels (e.g. detail.comp's
    'NOT packed into u_heroes.w'), and a comment mentioning a uniform the variant
    does not declare would be a false positive. GLSL has no string literals, so
    naive stripping is exact."""
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub(" ", src))


def _guard_symbols(src: str) -> set[str]:
    return set(_GUARD_IFDEF.findall(src)) | set(_GUARD_DEFINED.findall(src))


def _undeclared_uniforms(projected: str) -> set[str]:
    """The ``u_*`` symbols this projection USES but does not DECLARE."""
    code = _strip_comments(projected)
    return set(_UNIFORM_USE.findall(code)) - set(_UNIFORM_DECL.findall(code))


@pytest.mark.parametrize(("pkg", "name"), sorted(_VARIANT_AXES))
def test_variant_axes_are_complete(pkg: str, name: str):
    """Anti-rot: _preprocess treats an unlisted ``#ifdef FOO`` as permanently OFF,
    so a new variant axis would silently drop out of the matrix below and its
    blocks would never be projected ON. Adding a define to one of these kernels
    must fail here until it is added to _VARIANT_AXES."""
    src, _ = _load_flattened(pkg, name, {})
    assert _guard_symbols(src) == set(_VARIANT_AXES[(pkg, name)]), (
        f"{name}: preprocessor guard symbols drifted from _VARIANT_AXES. Add the new "
        f"define to _VARIANT_AXES (and confirm _preprocess evaluates its directive form)."
    )


@pytest.mark.parametrize(("pkg", "name"), sorted(_VARIANT_AXES))
def test_every_variant_declares_the_uniforms_it_uses(pkg: str, name: str):
    """Every reachable define combination must declare every ``u_*`` it uses.

    Exhaustive over the cartesian product (detail.comp 8, derive.comp 64) rather
    than the combinations shipped by presets: the crash this guards against was
    reachable only from an unshipped combination, which is exactly why per-feature
    tests -- each exercising its own feature at the 'normal' combination -- missed it.
    """
    axes = _VARIANT_AXES[(pkg, name)]
    broken: dict[str, set[str]] = {}
    for on in itertools.product([False, True], repeat=len(axes)):
        defines = {ax for ax, enabled in zip(axes, on, strict=True) if enabled}
        src, _ = _load_flattened(pkg, name, dict.fromkeys(defines, "1"))
        missing = _undeclared_uniforms(_preprocess(src, defines))
        if missing:
            broken["+".join(sorted(defines)) or "<no defines>"] = missing
    assert not broken, (
        f"{name}: variant(s) use undeclared uniform(s) -> GLSL 'undefined variable' at "
        f"compile time for anyone who selects them: {broken}. A uniform must be declared "
        f"under the LOOSEST guard that any of its uses sit under."
    )

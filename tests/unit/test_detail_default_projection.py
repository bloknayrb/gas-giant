"""GPU-free drift guard for the default (no-defines) detail.comp program.

The SPREAD variant wraps several base-path lines in ``#ifdef SPREAD /
#else / #endif`` with the ``#else`` arm a VERBATIM copy of today's line. The GPU
routing byte-identity test proves the default OUTPUT is unchanged *now*; this
unit test locks it cheaply in CI (no GL context) going forward by evaluating the
NO-DEFINES preprocessor projection of the flattened source and hashing it. A
non-verbatim ``#else`` arm (or an accidental base-path edit) moves the hash.

``gl/context._load_flattened`` only expands ``#include``; it leaves the
``#ifdef`` directives as literal text (the GLSL compiler resolves them). So we
apply a minimal C-style preprocessor here for the exact directive subset the
kernels use (``#ifdef``/``#ifndef``/``#if defined(...)``/``#else``/``#endif``)."""
from __future__ import annotations

import hashlib

from gasgiant.gl.context import _load_flattened


def _eval_if(expr: str, defines: set[str]) -> bool:
    expr = expr.strip()
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

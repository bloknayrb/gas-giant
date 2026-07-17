"""GPU-free drift guard for the default (no-defines) derive.comp program.

The MASK variant (imported paint mask -> POST targets) wraps its uniforms and
its three target blocks in ``#ifdef MASK`` (and the emission target additionally
nests inside ``#ifdef EMISSION``). Those blocks preprocess entirely OUT of the
default program, so neutral-default output stays byte-identical -- the same
construction the EMISSION/CHROMA_FX variants use.

This locks that cheaply in CI (no GL context) by hashing the NO-DEFINES
preprocessor projection of the flattened derive.comp. Any leak of a MASK block
into the default path -- or an accidental base-path edit -- moves the hash. It
reuses the exact C-preprocessor projection helper the detail.comp guard defines.
T14/T17 extend this.
"""
from __future__ import annotations

import hashlib

# Reuse the identical directive-subset preprocessor from the sibling guard
# (prepend import mode puts tests/unit on sys.path); duplicating it would be a
# second source of truth for the same projection semantics.
from test_detail_default_projection import _preprocess

from gasgiant.gl.context import _load_flattened


def _no_defines_projection() -> str:
    source, _ = _load_flattened("gasgiant.render.kernels", "derive.comp", {})
    return _preprocess(source, set())


# Hash of the NO-DEFINES projection. Regenerate DELIBERATELY (paste the printed
# value) ONLY when a real base-path change to derive.comp is intended -- never to
# paper over a MASK block leaking into the default path.
GOLDEN = "808b9b2561f5410e79fef188f397d0bbebc19f39d142d1a0dc68cebfd2c8c054"


def test_default_program_projection_is_stable():
    digest = hashlib.sha256(_no_defines_projection().encode("utf-8")).hexdigest()
    assert digest == GOLDEN, digest


def test_mask_symbols_absent_from_default_projection():
    """Semantic check: the default projection must strip every MASK uniform."""
    proj = _no_defines_projection()
    for sym in (
        "u_mask",
        "u_mask_band_fade",
        "u_mask_emission_gain",
        "u_mask_detail_gain",
        "u_band_tint",
        "u_band_tint_strength",
    ):
        assert sym not in proj, f"{sym} leaked into the default (no-defines) program"


def test_projection_cube_symbols_absent_from_default_projection():
    """T17: the PROJECTION_CUBE variant (cube-face -> lat/lon mapping) wraps its
    uniform and the whole mapping in ``#ifdef PROJECTION_CUBE`` with a VERBATIM
    ``#else`` arm (the current equirect uv lines). The default (no-defines)
    projection must strip every cube-only token; the golden hash above additionally
    pins that the surviving equirect arm is byte-for-byte today's text."""
    proj = _no_defines_projection()
    for sym in ("u_cube_face", "cube_lat", "cube_lon", "PROJECTION_CUBE"):
        assert sym not in proj, f"{sym} leaked into the default (no-defines) program"


def test_projection_cube_symbols_present_in_cube_projection():
    """Sanity: forcing PROJECTION_CUBE compiles the face uniform + the cube-face
    direction -> (lat, lon) mapping in."""
    source, _ = _load_flattened("gasgiant.render.kernels", "derive.comp", {})
    proj = _preprocess(source, {"PROJECTION_CUBE"})
    for sym in ("u_cube_face", "cube_lat", "cube_lon"):
        assert sym in proj, f"{sym} missing from the PROJECTION_CUBE projection"


def test_mask_symbols_present_in_mask_projection():
    """Sanity: forcing MASK compiles the uniforms + the target blocks in."""
    source, _ = _load_flattened("gasgiant.render.kernels", "derive.comp", {})
    proj = _preprocess(source, {"MASK"})
    for sym in ("u_mask", "u_mask_band_fade", "u_mask_detail_gain", "mask_band_col"):
        assert sym in proj, f"{sym} missing from the MASK projection"


def test_band_tint_symbols_present_in_band_tint_projection():
    """Sanity: forcing BAND_TINT compiles the tint uniforms + sample block in."""
    source, _ = _load_flattened("gasgiant.render.kernels", "derive.comp", {})
    proj = _preprocess(source, {"BAND_TINT"})
    for sym in ("u_band_tint", "u_band_tint_strength", "tintColor"):
        assert sym in proj, f"{sym} missing from the BAND_TINT projection"


def test_detail_chroma_symbols_absent_from_default_projection():
    """appearance.detail_chroma: uniform + Oklab material push must strip from
    the default (no-defines) program -- byte-identity when off is by
    construction, not by hope."""
    proj = _no_defines_projection()
    for sym in ("u_detail_chroma", "DETAIL_CHROMA"):
        assert sym not in proj, f"{sym} leaked into the default (no-defines) program"


def test_detail_chroma_symbols_present_in_detail_chroma_projection():
    """Sanity both ways: (a) forcing DETAIL_CHROMA alone compiles the uniform,
    the push body, AND the oklab functions -- oklab.glsl arrives through ONE
    compound-guard include (``#if defined(CHROMA_FX) || defined(DETAIL_CHROMA)``)
    because the flattener's include-once guard is filename-keyed and
    #ifdef-blind: a second guarded #include would expand to nothing and this
    exact variant would fail to compile at runtime; (b) CHROMA_FX alone still
    gets the oklab functions after the hoist."""
    source, _ = _load_flattened("gasgiant.render.kernels", "derive.comp", {})
    proj = _preprocess(source, {"DETAIL_CHROMA"})
    for sym in ("u_detail_chroma", "srgb_to_oklab", "oklab_to_srgb"):
        assert sym in proj, f"{sym} missing from the DETAIL_CHROMA projection"
    proj_cfx = _preprocess(source, {"CHROMA_FX"})
    for sym in ("srgb_to_oklab", "u_chroma_scale"):
        assert sym in proj_cfx, f"{sym} missing from the CHROMA_FX projection post-hoist"

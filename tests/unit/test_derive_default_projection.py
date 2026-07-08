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
    ):
        assert sym not in proj, f"{sym} leaked into the default (no-defines) program"


def test_mask_symbols_present_in_mask_projection():
    """Sanity: forcing MASK compiles the uniforms + the target blocks in."""
    source, _ = _load_flattened("gasgiant.render.kernels", "derive.comp", {})
    proj = _preprocess(source, {"MASK"})
    for sym in ("u_mask", "u_mask_band_fade", "u_mask_detail_gain", "mask_band_col"):
        assert sym in proj, f"{sym} missing from the MASK projection"

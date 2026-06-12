"""Numpy port of the repo's AgX view-transform approximation.

Source of truth: src/gasgiant/app/shaders/agx.glsl (Benjamin Wrensch /
iolite, MIT) — the SAME approximation the GUI preview trusts, ported
line-for-line so the comparison metrics, the preview, and (approximately)
Blender agree. The GLSL mat3 constructors are COLUMN-major; the arrays
below are the transposes of the literals as written in the shader. Note
the repo's agxEotf deliberately omits upstream's final pow(2.2).

Honesty caveats, stated rather than discovered:
- The reference photographs (e.g. PIA07782) are display-referred products
  whose rendering curve is NOT AgX; comparing agx_view(ours) against them
  is display-vs-display — strictly less wrong than raw-vs-ref, not "right".
- This polynomial approximation is not bit-exact Blender AgX (no outset);
  error is largest in saturated highlights.
- Our map is albedo and the Blender render is lit. Measured sensitivity of
  Oklab-chroma retention to the demo-scene sun exposure is negligible for
  belts and small for zones, so no exposure factor is applied.

numpy-only, no gasgiant.params dependency (same contract as gradient.py).
"""

from __future__ import annotations

import hashlib

import numpy as np

# Transposes of the GLSL column-major mat3 literals in agx.glsl.
_AGX_MAT = np.array(
    [
        [0.842479062253094, 0.0784335999999992, 0.0792237451477643],
        [0.0423282422610123, 0.878468636469772, 0.0791661274605434],
        [0.0423756549057051, 0.0784336, 0.879142973793104],
    ],
    dtype=np.float64,
)
_AGX_MAT_INV = np.array(
    [
        [1.19687900512017, -0.0980208811401368, -0.0990297440797205],
        [-0.0528968517574562, 1.15190312990417, -0.0989611768448433],
        [-0.0529716355144438, -0.0980434501171241, 1.15107367264116],
    ],
    dtype=np.float64,
)
_MIN_EV = -12.47393
_MAX_EV = 4.026069
_POLY = (15.5, -40.14, 31.96, -6.868, 0.4298, 0.1191, -0.00232)  # x^6 .. x^0


def _contrast_approx(x: np.ndarray) -> np.ndarray:
    x2 = x * x
    x4 = x2 * x2
    c6, c5, c4, c3, c2, c1, c0 = _POLY
    return c6 * x4 * x2 + c5 * x4 * x + c4 * x4 + c3 * x2 * x + c2 * x2 + c1 * x + c0


def agx_view(srgb: np.ndarray) -> np.ndarray:
    """(..., 3) display sRGB in [0, 1] -> AgX-approximation view, float32.

    Mirrors agx.glsl viewTransform(c, 1): srgbToLinear -> agx_mat ->
    log2 clamp [min_ev, max_ev] -> normalize -> 6th-order contrast
    polynomial -> agx_mat_inv. Inputs are clipped to [0, 1]; the output is
    clipped to [0, 1] (the shader leaves that to the display)."""
    c = np.clip(np.asarray(srgb, dtype=np.float64), 0.0, 1.0)
    lin = np.where(c >= 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    val = lin @ _AGX_MAT.T
    with np.errstate(divide="ignore"):
        val = np.clip(np.log2(val), _MIN_EV, _MAX_EV)
    val = (val - _MIN_EV) / (_MAX_EV - _MIN_EV)
    val = _contrast_approx(val)
    out = val @ _AGX_MAT_INV.T
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def agx_constants_checksum() -> str:
    """Short hash of the ported constants — recorded with baselines so a
    later change to this module visibly invalidates them."""
    blob = b"".join(
        (
            _AGX_MAT.tobytes(),
            _AGX_MAT_INV.tobytes(),
            np.float64([_MIN_EV, _MAX_EV, *_POLY]).tobytes(),
        )
    )
    return hashlib.sha1(blob).hexdigest()[:12]


def quartile_chroma_retention(img: np.ndarray, bins: int = 30, lat_max: float = 50.0) -> dict:
    """Per-quartile AgX Oklab-chroma retention of THIS image: mean of
    (chroma after agx_view) / (chroma raw) over |lat| <= lat_max bins.

    Retention is color-dependent, so a table measured on one palette is
    stale once the palette moves — re-measure from the current render at
    tuning time rather than reusing recorded numbers."""
    from gasgiant.palette.reference import latitude_profile

    p_raw = latitude_profile(img, bins)
    p_agx = latitude_profile(agx_view(img), bins)
    mask = np.abs(p_raw.lat_deg) <= lat_max
    eps = 1e-6
    return {
        "belt": float((p_agx.belt_chroma[mask] / (p_raw.belt_chroma[mask] + eps)).mean()),
        "zone": float((p_agx.zone_chroma[mask] / (p_raw.zone_chroma[mask] + eps)).mean()),
    }

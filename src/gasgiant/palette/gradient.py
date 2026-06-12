"""Bake gradient stops into a lookup table.

Takes plain (pos, (r, g, b)) tuples — deliberately not the pydantic stop model,
so this layer has no dependency on gasgiant.params.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Stop = tuple[float, tuple[float, float, float]]


def bake_lut(stops: Sequence[Stop], size: int = 256) -> np.ndarray:
    """(size, 4) float32 RGBA LUT, linearly interpolated, clamped past the ends."""
    if not stops:
        raise ValueError("at least one gradient stop required")
    ordered = sorted(stops, key=lambda s: s[0])
    pos = np.array([s[0] for s in ordered], dtype=np.float32)
    rgb = np.array([s[1] for s in ordered], dtype=np.float32)
    # LUT sample i is looked up at texel center (i + 0.5) / size
    x = (np.arange(size, dtype=np.float32) + 0.5) / size
    lut = np.empty((size, 4), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(x, pos, rgb[:, c])
    lut[:, 3] = 1.0
    return lut


Row = tuple[float, Sequence[Stop]]  # (signed latitude in degrees, gradient stops)


def bake_rows(rows: Sequence[Row], size: int = 256, height: int = 64) -> np.ndarray:
    """(height, size, 4) float32 LUT: each output row is the palette at one
    uniform latitude, row 0 at -90 deg (texture v=0), texel-center convention.

    Anchor rows are baked with bake_lut and blended across latitude with a
    smoothstep weight (piecewise-linear weights leave visible C1 creases at
    the anchors) in Oklab (sRGB-space lerps between e.g. blue-gray and ochre
    pass through muddy desaturated midpoints). Latitudes outside the
    outermost anchors clamp; a single row therefore reproduces bake_lut
    exactly on every output row.
    """
    if not rows:
        raise ValueError("at least one palette row required")
    ordered = sorted(rows, key=lambda r: r[0])
    lats = np.array([r[0] for r in ordered], dtype=np.float32)
    luts = [bake_lut(stops, size) for _, stops in ordered]

    out = np.empty((height, size, 4), dtype=np.float32)
    centers = -90.0 + (np.arange(height, dtype=np.float32) + 0.5) / height * 180.0
    for i, lat in enumerate(centers):
        j = int(np.searchsorted(lats, lat)) - 1
        if j < 0 or len(ordered) == 1:
            out[i] = luts[0]
        elif j >= len(ordered) - 1:
            out[i] = luts[-1]
        else:
            t = (lat - lats[j]) / (lats[j + 1] - lats[j])
            w = t * t * (3.0 - 2.0 * t)
            if w <= 0.0 or np.array_equal(luts[j], luts[j + 1]):
                out[i] = luts[j]
            elif w >= 1.0:
                out[i] = luts[j + 1]
            else:
                blended = _oklab_to_srgb(
                    (1.0 - w) * _srgb_to_oklab(luts[j][:, :3])
                    + w * _srgb_to_oklab(luts[j + 1][:, :3])
                )
                out[i, :, :3] = blended
                out[i, :, 3] = 1.0
    return out


# -- Oklab (Ottosson) --------------------------------------------------------------

_LMS_FROM_LIN = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ],
    dtype=np.float64,
)
_OKLAB_FROM_LMS = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ],
    dtype=np.float64,
)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) sRGB in [0, 1] -> (..., 3) Oklab (L, a, b), float64."""
    lms = _srgb_to_linear(rgb.astype(np.float64)) @ _LMS_FROM_LIN.T
    return np.cbrt(lms) @ _OKLAB_FROM_LMS.T


_srgb_to_oklab = srgb_to_oklab


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ np.linalg.inv(_OKLAB_FROM_LMS).T) ** 3
    linear = lms @ np.linalg.inv(_LMS_FROM_LIN).T
    return np.clip(_linear_to_srgb(linear), 0.0, 1.0).astype(np.float32)

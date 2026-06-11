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

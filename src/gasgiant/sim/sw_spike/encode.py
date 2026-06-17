from __future__ import annotations

import numpy as np

from . import operators as ops
from .solver import SwState


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def to_tracer(st: SwState) -> np.ndarray:
    """Pack top-layer (thickness anomaly, ζ₁) into the render's RGBA tracer.

    r = color index  <- thickness anomaly (banded color)
    g = height       <- thickness (cloud altitude)
    b = detail       <- relative vorticity magnitude (the v1.6 hero morphology)
    a = tint         <- signed vorticity (storm polarity)
    """
    H, W = st.h1.shape
    h_anom = st.h1 - (st.h_eq1 if st.h_eq1 is not None else st.h1.mean())
    zeta_corner = ops.vorticity(st.u1, st.v1, st.g)          # (H+1, W)
    zeta_c = 0.5 * (zeta_corner[0:H] + zeta_corner[1:H + 1]) # (H, W) at centers
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = _norm(h_anom)
    rgba[..., 1] = _norm(st.h1)
    rgba[..., 2] = _norm(np.abs(zeta_c))
    rgba[..., 3] = _norm(zeta_c)
    return rgba

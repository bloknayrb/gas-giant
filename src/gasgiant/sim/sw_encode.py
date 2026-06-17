"""M3 production top-of-atmosphere encoder: 2-layer GPU state -> RGBA tracer.

Promotes ``sw_spike/encode.py``'s ``to_tracer`` to the production a-aware
2-layer state (``shallow_water_ref.Sw2State`` / the GPU download tuple).  The
top layer is the visible cloud deck; the lower layer is hidden.  We render the
TOP layer:

    r = color index  <- top-layer thickness anomaly (h1 - h_eq1)  (banded color)
    g = height       <- top-layer thickness h1                    (cloud altitude)
    b = detail       <- |zeta_1| relative-vorticity magnitude     (the v1.6 hero)
    a = tint         <- signed zeta_1                              (storm polarity)

Each channel is robustly normalized into [0, 1] (1st/99th percentile clip) so
it feeds the unchanged ``derive.comp`` render path the same way the validated
spike encoder did.  ``zeta_1`` (top-layer vorticity) is the primary detail /
contrast channel -- the emergent eddies are the morphology the blind panel
judges.
"""

from __future__ import annotations

import numpy as np

from . import shallow_water_ref as ref


def _norm(a: np.ndarray) -> np.ndarray:
    """Robust [0,1] normalization via 1st/99th percentile clip (matches spike)."""
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def to_tracer_fields(
    h1: np.ndarray,
    u1: np.ndarray,
    v1: np.ndarray,
    g: ref.Grid,
    h_eq1: np.ndarray | None = None,
) -> np.ndarray:
    """Pack the TOP-layer fields into the render's (H, W, 4) RGBA tracer.

    Parameters mirror the production 2-layer GPU download: ``h1`` (H,W),
    ``u1`` (H,W), ``v1`` (H+1,W), the planetary ``Grid`` ``g``, and an optional
    radiative-equilibrium target ``h_eq1`` (H,W).  When ``h_eq1`` is None the
    anomaly is taken vs the zonal-by-latitude... simply the global mean.
    """
    H, W = h1.shape
    h_anom = h1 - np.asarray(h_eq1, dtype=h1.dtype) if h_eq1 is not None else h1 - float(h1.mean())

    # Top-layer relative vorticity at corners -> cell centers.
    zeta_corner = ref.vorticity(u1, v1, g)               # (H+1, W)
    zeta_c = 0.5 * (zeta_corner[0:H] + zeta_corner[1:H + 1])  # (H, W)

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = _norm(h_anom)         # banded color index
    rgba[..., 1] = _norm(h1)             # cloud altitude
    rgba[..., 2] = _norm(np.abs(zeta_c))  # detail (eddy magnitude) -- hero channel
    rgba[..., 3] = _norm(zeta_c)         # signed vorticity (storm polarity)
    return rgba


def to_tracer(st: ref.Sw2State) -> np.ndarray:
    """Convenience wrapper: encode a production 2-layer ``Sw2State`` top layer."""
    return to_tracer_fields(st.h1, st.u1, st.v1, st.g, st.h_eq1)

"""Honest M3 render-gate metrics.

The old T9 gate measured top-layer eddy Rossby -- wrong layer, wrong field. The
coupling gate instead asks two falsifiable questions of the RENDERED disk:

1. latitude_concentration: is eddy variance concentrated at the baroclinically
   active mid-latitudes (vs v1.6's latitude-flat FBM)?  Ratio of mean per-row
   zonal-anomaly variance inside the active band to outside it.
2. highfreq_energy: is the natural filamentary texture preserved (not smoothed
   away)?  Mean squared discrete Laplacian of the luminance.
"""
from __future__ import annotations

import numpy as np


def _luminance(img: np.ndarray) -> np.ndarray:
    """Accept (H, W) or (H, W, C>=3); return a (H, W) float64 luminance."""
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img.astype(np.float64)


def latitude_concentration(img: np.ndarray,
                           active_deg: tuple[float, float] = (20.0, 55.0)) -> float:
    """Ratio of mean per-row eddy variance inside the active latitude band to
    outside it. ~1 for a latitude-flat field; >1 when eddies concentrate in the
    band."""
    lum = _luminance(img)
    H, _ = lum.shape
    eddy = lum - lum.mean(axis=1, keepdims=True)
    row_var = eddy.var(axis=1)
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    active = (np.abs(lat) >= active_deg[0]) & (np.abs(lat) <= active_deg[1])
    off = ~active
    a = float(row_var[active].mean()) if active.any() else 0.0
    o = float(row_var[off].mean()) if off.any() else 0.0
    return a / (o + 1e-12)


def highfreq_energy(img: np.ndarray) -> float:
    """Mean squared 4-neighbour discrete Laplacian of the luminance -- a proxy
    for filamentary/high-frequency texture energy."""
    lum = _luminance(img)
    lap = (-4.0 * lum
           + np.roll(lum, 1, 0) + np.roll(lum, -1, 0)
           + np.roll(lum, 1, 1) + np.roll(lum, -1, 1))
    return float(np.mean(lap ** 2))

"""Honest M3 render-gate metrics.

The old T9 gate measured top-layer eddy Rossby -- wrong layer, wrong field. The
coupling gate instead asks falsifiable questions of the RENDERED disk:

1. banded_coherent_fraction (the HERO metric): of the eddy energy in the active
   mid-latitude band, what FRACTION sits in coherent low zonal wavenumbers
   (m=1..m_hi)?  The coupling's physical claim is that it adds COHERENT low-m
   eddy structure (festoons/roll-ups) at the active bands, so this fraction must
   RISE vs the v1.6 baseline. (Raw variance is the wrong gate: coherent organized
   waves carry LESS row-variance than the broadband filament hash they replace,
   so latitude_concentration paradoxically DROPS when fidelity improves.)
2. highfreq_energy: is the natural filamentary texture preserved (not smoothed
   away)?  Mean squared discrete Laplacian of the luminance.
3. latitude_concentration: a secondary broadband diagnostic (ratio of per-row
   eddy variance inside the active band to outside) -- reported, not gated.
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


def banded_coherent_fraction(img: np.ndarray,
                             active_deg: tuple[float, float] = (20.0, 55.0),
                             m_hi: int = 12) -> float:
    """Fraction of the active-band eddy energy carried by COHERENT low zonal
    wavenumbers (m=1..m_hi). For each active-band row: zonal eddy = row - row
    mean, power spectrum via rFFT; sum power in m=1..m_hi divided by total
    (m>=1) power. Averaged over active rows. A pure low-m wave -> ~1; broadband
    noise -> ~m_hi/(W/2). Rises when the coupling injects coherent banded
    structure; this is the metric that matches the physical claim."""
    lum = _luminance(img)
    H, W = lum.shape
    lat = 90.0 - (np.arange(H) + 0.5) / H * 180.0
    active = (np.abs(lat) >= active_deg[0]) & (np.abs(lat) <= active_deg[1])
    if not active.any():
        return 0.0
    rows = lum[active]
    eddy = rows - rows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(eddy, axis=1)) ** 2  # (n_active, W//2 + 1)
    hi = min(m_hi, power.shape[1] - 1)
    coherent = power[:, 1:hi + 1].sum(axis=1)
    total = power[:, 1:].sum(axis=1) + 1e-12
    return float(np.mean(coherent / total))


def highfreq_energy(img: np.ndarray) -> float:
    """Mean squared 4-neighbour discrete Laplacian of the luminance -- a proxy
    for filamentary/high-frequency texture energy."""
    lum = _luminance(img)
    lap = (-4.0 * lum
           + np.roll(lum, 1, 0) + np.roll(lum, -1, 0)
           + np.roll(lum, 1, 1) + np.roll(lum, -1, 1))
    return float(np.mean(lap ** 2))

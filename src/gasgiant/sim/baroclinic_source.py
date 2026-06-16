"""Coherent baroclinic vorticity source for M3 coupling.

Productionizes the spike-validated recipe (scripts/sw_m3_spike_coupling2.py):
derive a COHERENT geostrophic vorticity proxy from the eddy interface thickness
h2e = h2 - zonal_mean(h2) -- NOT from the raw relative vorticity, which is the
C-grid 2dx checkerboard (dominant zonal m~44-51). The coherent baroclinic signal
(m~5) lives in the interface thickness.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# Validated unstable baroclinic config (the brief's crux config). These name the
# grid + physics the source is BUILT from; they are consumed by the driver and
# gate script that spin the baroclinic solver (baroclinic_driver, sw_m3_couple),
# NOT by the functions below -- those read gp2 from the state object (st.gp2).
SRC_W, SRC_H = 192, 96
GP1, GP2, XI = 0.05, 0.3, 3.0

# Coherence gate: a usable source's dominant zonal wavenumber must be low.
M_GATE_MAX = 15


def dominant_zonal_m(field2d: np.ndarray,
                     row_frac: tuple[float, float] = (0.20, 0.42)) -> tuple[int, np.ndarray]:
    """Dominant zonal wavenumber of a (H, W) field: FFT a band of mid-latitude
    rows, average the power spectra, return argmax m (excluding DC) + spectrum."""
    H, _ = field2d.shape
    r0, r1 = int(row_frac[0] * H), int(row_frac[1] * H)
    rows = field2d[r0:r1]
    rows = rows - rows.mean(axis=1, keepdims=True)
    spec = (np.abs(np.fft.rfft(rows, axis=1)) ** 2).mean(axis=0)
    m = int(np.argmax(spec[1:]) + 1)
    return m, spec


def _smooth_periodic(field2d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth: wrap in longitude (axis=1), reflect in latitude (axis=0)."""
    return gaussian_filter(field2d, sigma=sigma, mode=("reflect", "wrap"))


def geostrophic_vorticity_source(st, smooth_sigma: float = 2.5,
                                 lat_band: tuple[float, float] = (10.0, 80.0),
                                 taper: float = 8.0) -> np.ndarray:
    """Coherent geostrophic vorticity proxy from the eddy interface thickness.

    zeta = (gp2 / f) * laplacian(smooth(h2 - zonal_mean(h2))), masked to the
    active mid-latitude band [lat_band] with a cosine taper (kills the np.gradient
    edge terms and 1/cos^2 zonal-Laplacian blow-up near the poles).
    Returns a (H, W) array on the baroclinic grid.
    """
    g = st.g
    h2e = st.h2 - st.h2.mean(axis=1, keepdims=True)
    h2e_s = _smooth_periodic(h2e, smooth_sigma)

    f_c = 2.0 * st.omega * np.sin(g.phi_c)
    f_safe = np.where(np.abs(f_c) < 1e-12, np.sign(f_c + 1e-30) * 1e-12, f_c)[:, None]
    d2_dphi2 = np.gradient(np.gradient(h2e_s, g.phi_c, axis=0), g.phi_c, axis=0)
    d2_dlam2 = np.gradient(np.gradient(h2e_s, g.dlam, axis=1), g.dlam, axis=1)
    cos2 = g.cos_c[:, None] ** 2 + 1e-12
    lap = (d2_dphi2 + d2_dlam2 / cos2) / (g.a ** 2)
    zeta = (st.gp2 / f_safe) * lap
    zeta = _smooth_periodic(zeta, 1.0)

    latdeg = np.degrees(g.phi_c)
    lo, hi = lat_band
    a_lat = np.clip((np.abs(latdeg) - lo) / taper, 0.0, 1.0)
    b_lat = np.clip((hi - np.abs(latdeg)) / taper, 0.0, 1.0)
    band = (0.5 - 0.5 * np.cos(np.pi * a_lat)) * (0.5 - 0.5 * np.cos(np.pi * b_lat))
    return zeta * band[:, None]


def assert_coherent(field2d: np.ndarray) -> int:
    """Reject a checkerboard source. Returns the dominant zonal m (<= M_GATE_MAX)."""
    m, _ = dominant_zonal_m(field2d)
    if m > M_GATE_MAX:
        raise ValueError(
            f"source dominant zonal m={m} exceeds coherence gate {M_GATE_MAX} "
            f"(checkerboard / grid-scale source, not a coherent eddy)"
        )
    return m


def resample_to_equirect(field2d: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    """Bilinear-resample (H, W) -> (grid_h, grid_w) and normalize to unit std,
    so a coupling `gain` is interpreted as a fraction of the solver Coriolis
    scale (coriolis_f0 = 3.0)."""
    zy = grid_h / field2d.shape[0]
    zx = grid_w / field2d.shape[1]
    resamp = zoom(field2d.astype(np.float32), (zy, zx), order=1, mode="nearest")
    resamp = resamp[:grid_h, :grid_w]
    std = float(np.std(resamp))
    if std > 0:
        resamp = resamp / std
    return np.ascontiguousarray(resamp.astype(np.float32))

"""Coherent baroclinic vorticity source for M3 coupling.

Productionizes the spike-validated recipe (scripts/sw_m3_spike_coupling2.py):
derive a COHERENT geostrophic vorticity proxy from the eddy interface thickness
h2e = h2 - zonal_mean(h2) -- NOT from the raw relative vorticity, which is the
C-grid 2dx checkerboard (dominant zonal m~44-51). The coherent baroclinic signal
lives in the interface thickness.

Eddy scale: the emergent dominant zonal wavenumber is set by the deformation
radius L_d (k_d^2 = 4*f0^2/(gp2*H)). gp2=0.075 puts the most-unstable mode at
m~14 (smaller, Jupiter-like mid-latitude storms). The earlier gp2=0.3 sat at
predicted m~7 (emergent ~m8) but was actually near-outcropping (broad,
incoherent blobs that blew up
~step 12.3k); the lower gp2 is BOTH finer-scale AND more coherent + stable
(single-mode share ~0.76 vs ~0.32, no outcrop through the coupled run). See
scripts/baro_scale_sweep.py for the CPU crux sweep that selected it.
"""
from __future__ import annotations

import cv2
import numpy as np

# Validated unstable baroclinic config. These name the grid + physics the source
# is BUILT from; they are consumed by the driver and gate script that spin the
# baroclinic solver (baroclinic_driver, sw_m3_couple), NOT by the functions below
# -- those read gp2 from the state object (st.gp2).
SRC_W, SRC_H = 192, 96
GP1, GP2, XI = 0.05, 0.075, 3.0
# Seed the instability at its predicted K_max (~m14 for this gp2) and smooth the
# resampled source on the smaller feature size (~SRC_W/m ~ 14px) so the resample
# does not blur the eddies away. Both consumed by baroclinic_driver.
M_ZONAL = 14
SMOOTH_SIGMA = 1.26

# Coherence gate: a usable source's dominant zonal wavenumber must be low (reject
# the C-grid checkerboard, m~44-51). Raised 15->20 to give the m~14 production
# mode session/seed margin while still rejecting grid-scale sources.
M_GATE_MAX = 20


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


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Normalized 1-D Gaussian, identical to scipy.ndimage's default kernel
    (truncate=4.0 -> radius = int(4*sigma + 0.5), weights exp(-x^2/2sigma^2),
    sum-normalized). Kept exactly scipy-shaped so _smooth_periodic is a drop-in
    scipy-free replacement (parity pinned in test_baroclinic_source)."""
    radius = int(4.0 * float(sigma) + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / float(sigma)) ** 2)
    return k / k.sum()


def _smooth_periodic(field2d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth: wrap in longitude (axis=1), reflect in latitude (axis=0).

    scipy-free equivalent of gaussian_filter(mode=("reflect", "wrap")): pad each
    axis by the kernel radius with its boundary rule (np.pad "symmetric" is
    scipy's "reflect"), run the separable filter with cv2 (a main dependency),
    and crop the padding — every kept pixel's footprint lies inside the padded
    array, so cv2's own border handling never contributes."""
    k = _gaussian_kernel1d(sigma)
    r = (k.size - 1) // 2
    if r == 0:
        return field2d.astype(np.float64, copy=True)
    padded = np.pad(field2d.astype(np.float64, copy=False), ((0, 0), (r, r)), mode="wrap")
    padded = np.pad(padded, ((r, r), (0, 0)), mode="symmetric")
    smoothed = cv2.sepFilter2D(padded, -1, k, k)
    return smoothed[r:-r, r:-r]


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


class IncoherentSourceError(ValueError):
    """The derived source failed the coherence gate (its dominant zonal mode is
    grid-scale / checkerboard, not a coherent eddy). Subclasses ValueError so
    existing `except ValueError` / pytest.raises(ValueError) sites keep working,
    while letting the coupling controller catch this *expected* degrade signal
    distinctly from an unrelated ValueError raised by a genuine bug."""


def assert_coherent(field2d: np.ndarray) -> int:
    """Reject a checkerboard source. Returns the dominant zonal m (<= M_GATE_MAX)."""
    m, _ = dominant_zonal_m(field2d)
    if m > M_GATE_MAX:
        raise IncoherentSourceError(
            f"source dominant zonal m={m} exceeds coherence gate {M_GATE_MAX} "
            f"(checkerboard / grid-scale source, not a coherent eddy)"
        )
    return m


def _zoom_bilinear(field2d: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """scipy-free equivalent of ndimage.zoom(order=1, grid_mode=False,
    mode="nearest"): align-corners bilinear — output index i samples input
    coordinate i*(in-1)/(out-1), so all coordinates stay in-domain and the
    "nearest" out-of-bounds rule never fires. cv2.resize is NOT a twin (it uses
    half-pixel-center mapping), so this is plain vectorized numpy; weights are
    computed in float64 and the result cast to the input dtype, matching
    scipy's internal double-precision spline path (parity pinned in
    test_baroclinic_source)."""
    in_h, in_w = field2d.shape
    ys = np.linspace(0.0, in_h - 1.0, out_h) if out_h > 1 else np.zeros(1)
    xs = np.linspace(0.0, in_w - 1.0, out_w) if out_w > 1 else np.zeros(1)
    y0 = np.clip(np.floor(ys).astype(np.intp), 0, in_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.intp), 0, in_w - 1)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    f = field2d.astype(np.float64, copy=False)
    top = f[np.ix_(y0, x0)] * (1.0 - wx) + f[np.ix_(y0, x1)] * wx
    bot = f[np.ix_(y1, x0)] * (1.0 - wx) + f[np.ix_(y1, x1)] * wx
    return (top * (1.0 - wy) + bot * wy).astype(field2d.dtype)


def resample_to_equirect(field2d: np.ndarray, grid_w: int, grid_h: int) -> np.ndarray:
    """Bilinear-resample (H, W) -> (grid_h, grid_w) and normalize to unit std,
    so a coupling `gain` is interpreted as a fraction of the solver Coriolis
    scale (coriolis_f0 = 3.0)."""
    resamp = _zoom_bilinear(field2d.astype(np.float32), grid_h, grid_w)
    std = float(np.std(resamp))
    if std > 0:
        resamp = resamp / std
    return np.ascontiguousarray(resamp.astype(np.float32))

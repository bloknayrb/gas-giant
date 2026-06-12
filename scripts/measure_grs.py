"""GRS measurement utilities for Gas Giant Studio v1.5 (P5 integration).

Three metrics that Fable calls on real renders to assess GRS collar quality:

  ring_ripple_std   — GRS-2: detect concentric-ring artifact via azimuthal
                      mean profile; high for theta-independent rings, low for
                      azimuthally modulated (fixed) rings.
  fit_ellipse_aspect — GRS-1: recover the spot's ellipse aspect ratio from
                       the dark perimeter ring via cv2.fitEllipse.
  ring_closure      — GRS-1b: detect incomplete / broken dark ring segments.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bilinear_sample(L: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly interpolate 2-D array L at fractional coordinates (xs, ys).

    Out-of-bounds samples are clamped to the nearest edge pixel.
    xs: column coords (float), ys: row coords (float), both shape (N,) or (R,C).
    Returns array of same shape as xs.
    """
    h, w = L.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    # Clamp
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    # Fractional offsets
    tx = (xs - x0).astype(np.float32)
    ty = (ys - y0).astype(np.float32)
    # Four corner values
    f00 = L[y0c, x0c].astype(np.float32)
    f10 = L[y0c, x1c].astype(np.float32)
    f01 = L[y1c, x0c].astype(np.float32)
    f11 = L[y1c, x1c].astype(np.float32)
    return (f00 * (1 - tx) * (1 - ty)
            + f10 * tx * (1 - ty)
            + f01 * (1 - tx) * ty
            + f11 * tx * ty)


def _moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average with edge clamping (reflect-pad then trim)."""
    pad = win // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(win, dtype=np.float64) / win
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad: pad + len(arr)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ring_ripple_std(
    L: np.ndarray,
    cx: float,
    cy: float,
    rc_px: float,
    aspect: float,
    q_lo: float = 1.0,
    q_hi: float = 1.85,
    n_bins: int = 24,
    n_theta: int = 180,
) -> float:
    """GRS-2 metric: azimuthal-mean ripple strength in the collar annulus.

    Algorithm:
    - Build a polar sample grid (n_bins × n_theta) over q ∈ [q_lo, q_hi] and
      theta ∈ [0, 2π) in the spot's elliptical frame (x-axis scaled by aspect).
    - Bilinearly interpolate L at each sample point.
    - Compute the azimuthal mean per radial bin to get a 1-D profile.
    - Detrend by subtracting a 5-bin centered moving average (removes broad
      Gaussians while preserving q*28 ripple, period ≈ 0.22 in q ≈ 6 bins).
    - Return std(residual): HIGH for concentric (theta-independent) rings,
      LOW for azimuthally modulated rings whose mean is ≈ 0.

    Parameters
    ----------
    L       : 2-D float luminance array (H × W).
    cx, cy  : hero center in pixels (float).
    rc_px   : core radius in pixels.
    aspect  : lon:lat elongation (ellipse x-semi-axis = aspect × rc_px).
    q_lo, q_hi : radial range in elliptical q units.
    n_bins  : number of radial stations.
    n_theta : number of angular stations per bin.
    """
    L = np.asarray(L, dtype=np.float32)

    q_vals = np.linspace(q_lo, q_hi, n_bins)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    # Build full grid (n_bins, n_theta)
    Q, TH = np.meshgrid(q_vals, theta_vals, indexing="ij")  # (n_bins, n_theta)
    xs = cx + Q * rc_px * aspect * np.cos(TH)
    ys = cy + Q * rc_px * np.sin(TH)

    samples = _bilinear_sample(L, xs, ys)  # (n_bins, n_theta)

    # Azimuthal mean per radial bin
    profile = samples.mean(axis=1)  # (n_bins,)

    # Detrend: subtract 5-bin centered moving average
    smooth = _moving_average(profile, 5)
    residual = profile - smooth

    return float(np.std(residual))


def fit_ellipse_aspect(
    L: np.ndarray,
    cx: float,
    cy: float,
    rc_px: float,
    search: float = 2.2,
) -> float | None:
    """GRS-1 integrity check: recover the dark-ring aspect ratio.

    Algorithm:
    - Mask a circular search region of radius search*rc_px around (cx, cy).
    - Within a narrow annular band near q ≈ 1 (0.7 ≤ q ≤ 1.3 using simple
      circular distance as aspect is unknown here), collect pixels below the
      20th-percentile luminance as "dark ring" candidates.
    - Fit an ellipse to those dark pixels via cv2.fitEllipse.
    - Return major_axis / minor_axis.  Returns None if fewer than 5 points.

    Parameters
    ----------
    L      : 2-D float luminance array.
    cx, cy : hero center in pixels.
    rc_px  : core radius in pixels.
    search : search radius in units of rc_px.
    """
    L = np.asarray(L, dtype=np.float32)
    h, w = L.shape

    # Pixel coordinate grids
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    dr = rows - cy
    dc = cols - cx
    dist = np.hypot(dr, dc)  # circular distance

    # Search region: circular disk of radius search*rc_px (aspect unknown here)
    search_mask = dist <= search * rc_px
    # Exclude the interior core (< 0.5 rc_px) to focus on perimeter
    search_mask &= dist >= 0.5 * rc_px

    if search_mask.sum() < 10:
        return None

    vals_in_region = L[search_mask]
    threshold = float(np.percentile(vals_in_region, 20))
    dark_mask = search_mask & (threshold >= L)

    pts = np.column_stack(np.where(dark_mask))  # (N, 2) as (row, col)
    if len(pts) < 5:
        return None

    # cv2.fitEllipse wants (N, 1, 2) in (x, y) = (col, row) order
    pts_xy = pts[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
    ellipse = cv2.fitEllipse(pts_xy)
    axes = ellipse[1]  # (width, height) = (2*semi_a, 2*semi_b)
    major = float(max(axes))
    minor = float(min(axes))
    if minor < 1e-6:
        return None
    return major / minor


def ring_closure(
    L: np.ndarray,
    cx: float,
    cy: float,
    rc_px: float,
    aspect: float,
    n_theta: int = 180,
) -> tuple[float, float]:
    """GRS-1b: measure dark-ring closure and depth uniformity.

    Algorithm:
    - Walk n_theta evenly-spaced rays from the hero center.
    - On each ray, sample L at 40 elliptical-q stations in [0.7, 1.3] using
      the spot's aspect to stretch the x-axis.
    - Find the L minimum along each ray; record its depth relative to the
      local maximum within that window.
    - A ray is "closed" (ring detected) if its depth exceeds a threshold
      (0.05 of the global L range in the region, floored at 0.02).
    - max_gap_deg: largest contiguous span of rays where no ring is detected.
    - depth_ratio: min ring depth / median ring depth around the full ring.
    Returns (max_gap_deg, depth_ratio).

    Fable's pass criteria: max_gap_deg < 45, depth_ratio >= 0.40.

    Parameters
    ----------
    L       : 2-D float luminance array.
    cx, cy  : hero center in pixels.
    rc_px   : core radius in pixels.
    aspect  : ellipse x-stretch factor.
    n_theta : number of angular rays.
    """
    L = np.asarray(L, dtype=np.float32)
    n_q = 40
    q_vals = np.linspace(0.7, 1.3, n_q)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    depths = np.zeros(n_theta, dtype=np.float64)
    for i, th in enumerate(theta_vals):
        xs = cx + q_vals * rc_px * aspect * np.cos(th)
        ys = cy + q_vals * rc_px * np.sin(th)
        ray = _bilinear_sample(L, xs, ys)
        local_max = float(ray.max())
        local_min = float(ray.min())
        depths[i] = local_max - local_min

    # Global depth threshold
    L_range = float(L.max() - L.min())
    threshold = max(0.02, 0.05 * L_range)

    detected = depths >= threshold

    # max gap: longest run of False in circular array
    if detected.all():
        max_gap_deg = 0.0
    elif not detected.any():
        max_gap_deg = 360.0
    else:
        # Double the array to handle wrap-around
        doubled = np.concatenate([detected, detected])
        max_gap = 0
        cur_gap = 0
        for val in doubled:
            if not val:
                cur_gap += 1
                if cur_gap > max_gap:
                    max_gap = cur_gap
            else:
                cur_gap = 0
        # Cap at n_theta (full circle)
        max_gap = min(max_gap, n_theta)
        max_gap_deg = float(max_gap) * 360.0 / n_theta

    # Depth ratio: min / median of detected rays (or 0 if fewer than 2)
    detected_depths = depths[detected]
    if len(detected_depths) < 2:
        depth_ratio = 0.0
    else:
        depth_ratio = float(detected_depths.min() / (np.median(detected_depths) + 1e-12))

    return (max_gap_deg, depth_ratio)

"""Per-latitude color statistics of an equirectangular (cylindrical) image.

The calibration ground truth for these statistics is a *cylindrical* map
(e.g. Cassini's PIA07782); globe photographs would need limb-darkening
removal and disk-to-latitude projection and are not supported here.

Means are deliberately avoided: the mean at a belt latitude averages the
belt, white ovals, dark rims, and streaks into a muddy mid-tone and would
calibrate a low-contrast palette. Instead each latitude bin is decomposed
by luminance quartile — the median color of the brightest quartile tracks
the zone/bright-cloud component, the darkest quartile the belt component —
and the 5th/95th luminance percentiles give the local contrast envelope.

numpy-only, plain arrays in/out: like gradient.py, this module takes no
dependency on gasgiant.params.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rec. 709 luma weights.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class LatitudeProfile:
    """All arrays are indexed by latitude bin, north (+90) first."""

    lat_deg: np.ndarray      # (bins,) bin-center latitude, degrees
    zone_rgb: np.ndarray     # (bins, 3) median color of the brightest luminance quartile
    belt_rgb: np.ndarray     # (bins, 3) median color of the darkest luminance quartile
    median_rgb: np.ndarray   # (bins, 3) overall median color
    lum_p5: np.ndarray       # (bins,) 5th percentile luminance
    lum_p95: np.ndarray      # (bins,) 95th percentile luminance
    lum_std: np.ndarray      # (bins,) luminance standard deviation

    @property
    def contrast(self) -> np.ndarray:
        """(bins,) local contrast envelope: p95 − p5 luminance."""
        return self.lum_p95 - self.lum_p5


def _quartile_median_rgb(rgb: np.ndarray, lum: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo, hi = np.quantile(lum, [lo_q, hi_q])
    sel = (lum >= lo) & (lum <= hi)
    return np.median(rgb[sel], axis=0)


def latitude_profile(img: np.ndarray, bins: int = 90) -> LatitudeProfile:
    """Profile an (H, W, 3) float image in [0, 1], assumed equirectangular
    with row 0 at +90 deg latitude (texel-center convention)."""
    h = img.shape[0]
    rows_per_bin = np.array_split(np.arange(h), bins)
    n = len(rows_per_bin)

    lat = np.empty(n, dtype=np.float32)
    zone = np.empty((n, 3), dtype=np.float32)
    belt = np.empty((n, 3), dtype=np.float32)
    med = np.empty((n, 3), dtype=np.float32)
    p5 = np.empty(n, dtype=np.float32)
    p95 = np.empty(n, dtype=np.float32)
    std = np.empty(n, dtype=np.float32)

    for i, rows in enumerate(rows_per_bin):
        lat[i] = 90.0 - (rows.mean() + 0.5) / h * 180.0
        rgb = img[rows].reshape(-1, 3).astype(np.float32)
        lum = rgb @ _LUMA
        zone[i] = _quartile_median_rgb(rgb, lum, 0.75, 1.0)
        belt[i] = _quartile_median_rgb(rgb, lum, 0.0, 0.25)
        med[i] = np.median(rgb, axis=0)
        p5[i], p95[i] = np.quantile(lum, [0.05, 0.95])
        std[i] = lum.std()

    return LatitudeProfile(lat, zone, belt, med, p5, p95, std)


def profile_distance(a: LatitudeProfile, b: LatitudeProfile) -> dict[str, float]:
    """Mean absolute differences between two profiles (resampled onto a's
    latitude grid). Keys: zone_rgb, belt_rgb, contrast."""
    out: dict[str, float] = {}

    def _resample(values: np.ndarray) -> np.ndarray:
        # np.interp wants ascending x; latitudes run +90 -> -90.
        if values.ndim == 1:
            return np.interp(a.lat_deg[::-1], b.lat_deg[::-1], values[::-1])[::-1]
        columns = [
            np.interp(a.lat_deg[::-1], b.lat_deg[::-1], values[::-1, c])[::-1]
            for c in range(values.shape[1])
        ]
        return np.stack(columns, axis=1)

    out["zone_rgb"] = float(np.abs(a.zone_rgb - _resample(b.zone_rgb)).mean())
    out["belt_rgb"] = float(np.abs(a.belt_rgb - _resample(b.belt_rgb)).mean())
    out["contrast"] = float(np.abs(a.contrast - _resample(b.contrast)).mean())
    return out

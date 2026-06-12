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

The chroma/lightness statistics are MEMBER statistics conditional on the
luminance quartile, in Oklab. Per-channel medians of a hue-spread
population regress toward gray, so the median *color* of a quartile can
match while its members are far more saturated — the quartile-conditional
member chroma sees that gap. Full-bin standard deviations would measure
the zone-belt separation, not within-band richness, which is why the
std/p95 statistics are quartile-conditional too.

numpy-only, plain arrays in/out: like gradient.py, this module takes no
dependency on gasgiant.params.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gasgiant.palette.gradient import _oklab_to_srgb, srgb_to_oklab

# Rec. 709 luma weights.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Below this mean member chroma a bin is effectively gray: hue direction is
# noise (JPEG chroma jitter dominates), so hue statistics are reported as 0.
_GRAY_CHROMA_FLOOR = 0.01


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
    zone_chroma: np.ndarray      # (bins,) median Oklab chroma over bright-quartile members
    belt_chroma: np.ndarray      # (bins,) median Oklab chroma over dark-quartile members
    zone_chroma_std: np.ndarray  # (bins,) member-chroma std within the bright quartile
    belt_chroma_std: np.ndarray  # (bins,) member-chroma std within the dark quartile
    zone_L_std: np.ndarray       # (bins,) member Oklab-L std within the bright quartile
    belt_L_std: np.ndarray       # (bins,) member Oklab-L std within the dark quartile
    belt_chroma_p95: np.ndarray  # (bins,) dark-quartile member-chroma 95th percentile
    hue_spread: np.ndarray       # (bins,) chroma-weighted circular hue spread, 0..1
    texture_energy: np.ndarray   # (bins,) mean |grad L| — fine-texture density proxy

    @property
    def contrast(self) -> np.ndarray:
        """(bins,) local contrast envelope: p95 − p5 luminance."""
        return self.lum_p95 - self.lum_p5


def _quartile_sel(lum: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo, hi = np.quantile(lum, [lo_q, hi_q])
    return (lum >= lo) & (lum <= hi)


def latitude_profile(img: np.ndarray, bins: int = 90) -> LatitudeProfile:
    """Profile an (H, W, 3) float image in [0, 1], assumed equirectangular
    with row 0 at +90 deg latitude (texel-center convention)."""
    h = img.shape[0]
    rows_per_bin = np.array_split(np.arange(h), bins)
    n = len(rows_per_bin)

    # One Oklab conversion for the whole image feeds every member statistic.
    lab = srgb_to_oklab(img.reshape(-1, 3)).reshape(img.shape[0], img.shape[1], 3)
    big_l = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    gy, gx = np.gradient(big_l)
    grad_l = np.hypot(gx, gy)

    lat = np.empty(n, dtype=np.float32)
    zone = np.empty((n, 3), dtype=np.float32)
    belt = np.empty((n, 3), dtype=np.float32)
    med = np.empty((n, 3), dtype=np.float32)
    p5 = np.empty(n, dtype=np.float32)
    p95 = np.empty(n, dtype=np.float32)
    std = np.empty(n, dtype=np.float32)
    z_c = np.empty(n, dtype=np.float32)
    b_c = np.empty(n, dtype=np.float32)
    z_cs = np.empty(n, dtype=np.float32)
    b_cs = np.empty(n, dtype=np.float32)
    z_ls = np.empty(n, dtype=np.float32)
    b_ls = np.empty(n, dtype=np.float32)
    b_cp = np.empty(n, dtype=np.float32)
    hue = np.empty(n, dtype=np.float32)
    tex = np.empty(n, dtype=np.float32)

    for i, rows in enumerate(rows_per_bin):
        lat[i] = 90.0 - (rows.mean() + 0.5) / h * 180.0
        rgb = img[rows].reshape(-1, 3).astype(np.float32)
        lum = rgb @ _LUMA
        ab = lab[rows].reshape(-1, 3)[:, 1:]
        c_bin = chroma[rows].reshape(-1)
        l_bin = big_l[rows].reshape(-1)

        sel_hi = _quartile_sel(lum, 0.75, 1.0)
        sel_lo = _quartile_sel(lum, 0.0, 0.25)
        zone[i] = np.median(rgb[sel_hi], axis=0)
        belt[i] = np.median(rgb[sel_lo], axis=0)
        med[i] = np.median(rgb, axis=0)
        p5[i], p95[i] = np.quantile(lum, [0.05, 0.95])
        std[i] = lum.std()

        z_c[i] = np.median(c_bin[sel_hi])
        b_c[i] = np.median(c_bin[sel_lo])
        z_cs[i] = c_bin[sel_hi].std()
        b_cs[i] = c_bin[sel_lo].std()
        z_ls[i] = l_bin[sel_hi].std()
        b_ls[i] = l_bin[sel_lo].std()
        b_cp[i] = np.quantile(c_bin[sel_lo], 0.95)

        # Chroma-weighted circular hue spread: 1 - |sum(a,b)| / sum|(a,b)|.
        # Weighting by chroma keeps near-neutral JPEG noise from dominating.
        norms = c_bin.sum()
        if c_bin.mean() < _GRAY_CHROMA_FLOOR or norms <= 0.0:
            hue[i] = 0.0
        else:
            hue[i] = 1.0 - float(np.linalg.norm(ab.sum(axis=0))) / float(norms)
        tex[i] = grad_l[rows].mean()

    return LatitudeProfile(
        lat, zone, belt, med, p5, p95, std,
        z_c, b_c, z_cs, b_cs, z_ls, b_ls, b_cp, hue, tex,
    )


def chroma_restored_rgb(
    rgb: np.ndarray,
    lum: np.ndarray,
    lo_q: float,
    hi_q: float,
    chroma_pct: float = 0.6,
) -> np.ndarray:
    """Quartile color with chroma restored to the chroma_pct percentile of
    the quartile MEMBERS. Median L and the median (a, b) direction give a
    robust lightness/hue; the chroma magnitude is re-inflated because
    per-channel medians of a hue-spread population regress toward gray.

    Restoring magnitude along the robust hue direction cannot latch onto a
    chromatic minority and flip hue (fitting the top-chroma sub-quartile
    could — e.g. festoon blue inside a belt). Guard: when the median member
    chroma is below the gray floor the hue direction is noise (polar grays)
    and the plain per-channel median is returned instead."""
    sel = _quartile_sel(lum, lo_q, hi_q)
    members = rgb[sel]
    lab = srgb_to_oklab(members)
    c_members = np.hypot(lab[:, 1], lab[:, 2])
    if np.median(c_members) < _GRAY_CHROMA_FLOOR:
        return np.median(members, axis=0)
    big_l = np.median(lab[:, 0])
    hue = np.arctan2(np.median(lab[:, 2]), np.median(lab[:, 1]))
    c = np.quantile(c_members, chroma_pct)
    lab_out = np.array([[big_l, c * np.cos(hue), c * np.sin(hue)]])
    return _oklab_to_srgb(lab_out)[0]


# Stop positions and the luminance windows each is fitted from (pixel-level
# anchor fits). 5 stops match the hand-extended factory rows' structure.
# The 5-stop ENDPOINT windows are luminance TAILS, not quartiles: storm
# stamps map T0 to the gradient endpoints, and endpoint stops fitted from
# windows overlapping the interior ones compress the very range the stamps
# live in (the v1.4 audit's invisible-ovals/pearls/barges root cause —
# tracer deltas were strong, the palette top end was flat).
STOP_WINDOWS: dict[int, tuple[tuple[float, tuple[float, float]], ...]] = {
    3: ((0.0, (0.0, 0.25)), (0.5, (0.375, 0.625)), (1.0, (0.75, 1.0))),
    5: (
        (0.0, (0.0, 0.12)),
        (0.25, (0.125, 0.375)),
        (0.5, (0.375, 0.625)),
        (0.75, (0.625, 0.875)),
        (1.0, (0.88, 1.0)),
    ),
}


def anchor_fit(
    img: np.ndarray,
    anchor_deg: float,
    window_deg: float,
    mode: str = "median",
    chroma_pct: float = 0.6,
    stops: int = 3,
) -> list[tuple[float, np.ndarray]]:
    """Fit (pos, rgb) gradient stops for one anchor latitude directly from
    the pixel rows within +-window_deg (latitude_profile only exposes
    per-bin aggregates; the chroma-restored fit needs member pixels)."""
    h = img.shape[0]
    lat = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    rows = np.abs(lat - anchor_deg) <= window_deg
    if not rows.any():
        nearest = np.argsort(np.abs(lat - anchor_deg))[: max(3, h // 30)]
        rows = np.zeros(h, dtype=bool)
        rows[nearest] = True
    rgb = img[rows].reshape(-1, 3).astype(np.float32)
    lum = rgb @ _LUMA
    out: list[tuple[float, np.ndarray]] = []
    for pos, (lo, hi) in STOP_WINDOWS[stops]:
        if mode == "chroma-restore":
            color = chroma_restored_rgb(rgb, lum, lo, hi, chroma_pct)
        else:
            color = np.median(rgb[_quartile_sel(lum, lo, hi)], axis=0)
        out.append((pos, color))
    return out


def expand_stop_span(
    fitted: list[tuple[float, np.ndarray]], min_span: float
) -> list[tuple[float, np.ndarray]]:
    """Expand a fitted row's Oklab-L span about its mean when the source
    window is blur-flattened. The reference map's poleward latitudes are
    projection-blurred (the quartile windows converge), which fits
    near-constant gradients that crush ALL T0 dynamic range downstream —
    banding, storm stamps, polar cyclones; PIA21641 shows the real polar
    dynamic range is anything but flat. Per-stop hue/chroma are preserved;
    only the L deviations scale. No-op when the span already clears the
    floor (low-latitude rows are untouched)."""
    colors = np.array([c for _, c in fitted], dtype=np.float32)
    lab = srgb_to_oklab(colors)
    span = float(lab[:, 0].max() - lab[:, 0].min())
    if span >= min_span or span <= 1e-6:
        return fitted
    mean_l = float(lab[:, 0].mean())
    lab[:, 0] = mean_l + (lab[:, 0] - mean_l) * (min_span / span)
    out = _oklab_to_srgb(lab)
    return [
        (pos, np.clip(out[i], 0.0, 1.0).astype(np.float32))
        for i, (pos, _) in enumerate(fitted)
    ]


def _moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average with edge replication (odd win). A moving
    MEDIAN would snap to the majority class inside bands wider than half the
    window (the EZ is ~20 deg) and zero the classification signal exactly
    where it matters; the mean interpolates instead."""
    half = win // 2
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    return np.convolve(padded, np.ones(win) / win, mode="valid")


def band_template_arrays(
    img: np.ndarray,
    max_bands: int = 18,
    min_width_deg: float = 2.0,
    lat_extent: float = 76.0,
) -> dict[str, np.ndarray]:
    """Extract a band skeleton from a cylindrical reference image.

    Per-row median luminance, smoothed (~1.5 deg), classified zone/belt
    against a wide (~45 deg) moving-mean baseline; class transitions give
    edge latitudes. Bands narrower than min_width_deg are merged 3-way into
    their (same-class) neighbors, weakest-first; the same merge trims the
    count to max_bands and forces an EVEN count (the `values < median`
    identity convention cannot represent belts-majority odd layouts).

    Returns plain arrays only — no dependency on gasgiant.params; callers
    map band luminance into color-index/height space themselves:
      edges_deg  (n+1,) descending from +90 to -90 (interior <= lat_extent)
      band_lum   (n,)   median smoothed row luminance per band
      is_zone    (n,)   bool, brighter-than-baseline class
    """
    h = img.shape[0]
    deg_per_row = 180.0 / h
    lat = 90.0 - (np.arange(h) + 0.5) * deg_per_row
    row_med = np.median(img.astype(np.float32) @ _LUMA, axis=1)
    win = max(3, int(round(1.5 / deg_per_row)) | 1)
    smooth = np.convolve(row_med, np.ones(win) / win, mode="same")
    base = _moving_mean(smooth, max(3 * win, int(round(45.0 / deg_per_row)) | 1))

    rows = np.where(np.abs(lat) <= lat_extent)[0]
    cls = smooth[rows] > base[rows]
    flips = np.where(cls[1:] != cls[:-1])[0]
    # Band spans as row-index ranges [start, stop) over `rows`.
    starts = np.concatenate([[0], flips + 1])
    stops = np.concatenate([flips + 1, [len(rows)]])

    def _lum(s: int, e: int) -> float:
        return float(np.median(smooth[rows[s:e]]))

    bands = [
        {"s": int(s), "e": int(e), "zone": bool(cls[s]), "lum": _lum(s, e)}
        for s, e in zip(starts, stops, strict=False)
    ]

    def _width(b: dict) -> float:
        return (b["e"] - b["s"]) * deg_per_row

    def _weakness(b: dict) -> float:
        span = rows[b["s"]:b["e"]]
        contrast = float(np.abs(smooth[span] - base[span]).mean())
        return _width(b) * max(contrast, 1e-6)

    def _merge_one(idx: int) -> None:
        # 3-way merge: neighbors share a class (alternation), so removing
        # band idx folds it and both neighbors into one band of their class.
        lo, hi = max(idx - 1, 0), min(idx + 1, len(bands) - 1)
        merged = {
            "s": bands[lo]["s"], "e": bands[hi]["e"],
            "zone": bands[lo]["zone"] if lo != idx else not bands[idx]["zone"],
            "lum": 0.0,
        }
        merged["lum"] = _lum(merged["s"], merged["e"])
        bands[lo:hi + 1] = [merged]

    while len(bands) > 2:
        narrow = [i for i, b in enumerate(bands) if _width(b) < min_width_deg]
        if narrow:
            _merge_one(min(narrow, key=lambda i: _weakness(bands[i])))
        elif len(bands) > max_bands:
            # Interior merge removes 2 bands and preserves parity.
            interior = range(1, len(bands) - 1)
            _merge_one(min(interior, key=lambda i: _weakness(bands[i])))
        elif len(bands) % 2 == 1:
            # Parity fix must be an EDGE merge (removes exactly 1 band).
            _merge_one(min((0, len(bands) - 1), key=lambda i: _weakness(bands[i])))
        else:
            break

    edges = [90.0]
    for prev, nxt in zip(bands[:-1], bands[1:], strict=False):
        edges.append(float(0.5 * (lat[rows[prev["e"] - 1]] + lat[rows[nxt["s"]]])))
    edges.append(-90.0)
    return {
        "edges_deg": np.asarray(edges, dtype=np.float64),
        "band_lum": np.asarray([b["lum"] for b in bands], dtype=np.float64),
        "is_zone": np.asarray([b["zone"] for b in bands], dtype=bool),
    }


MEDIAN_KEYS = ("zone_rgb", "belt_rgb", "contrast", "zone_chroma", "belt_chroma")
VARIANCE_KEYS = (
    "zone_chroma_std", "belt_chroma_std", "zone_L_std", "belt_L_std",
    "belt_chroma_p95", "hue_spread", "texture_energy",
)


def _resample_onto(a: LatitudeProfile, b: LatitudeProfile, values: np.ndarray) -> np.ndarray:
    # np.interp wants ascending x; latitudes run +90 -> -90.
    if values.ndim == 1:
        return np.interp(a.lat_deg[::-1], b.lat_deg[::-1], values[::-1])[::-1]
    columns = [
        np.interp(a.lat_deg[::-1], b.lat_deg[::-1], values[::-1, c])[::-1]
        for c in range(values.shape[1])
    ]
    return np.stack(columns, axis=1)


def profile_distance(a: LatitudeProfile, b: LatitudeProfile) -> dict[str, float]:
    """Mean absolute differences between two profiles (resampled onto a's
    latitude grid). Keys: the level statistics (zone_rgb, belt_rgb, contrast,
    zone_chroma, belt_chroma) and the variance statistics (quartile-
    conditional stds, belt_chroma_p95, hue_spread, texture_energy)."""
    out: dict[str, float] = {}
    for key in MEDIAN_KEYS + VARIANCE_KEYS:
        va = getattr(a, key)
        vb = getattr(b, key)
        out[key] = float(np.abs(va - _resample_onto(a, b, vb)).mean())
    return out


def profile_signed(
    a: LatitudeProfile, b: LatitudeProfile, lat_max: float | None = None
) -> dict[str, float]:
    """Mean SIGNED differences (a − b) for the chroma/lightness statistics —
    shows deficit vs overshoot direction. Optionally restricted to
    |lat| <= lat_max degrees (band latitudes)."""
    keys = (
        "zone_chroma", "belt_chroma", "zone_chroma_std", "belt_chroma_std",
        "zone_L_std", "belt_L_std", "belt_chroma_p95", "hue_spread",
        "texture_energy",
    )
    mask = np.ones_like(a.lat_deg, dtype=bool)
    if lat_max is not None:
        mask = np.abs(a.lat_deg) <= lat_max
    out: dict[str, float] = {}
    for key in keys:
        diff = getattr(a, key) - _resample_onto(a, b, getattr(b, key))
        out[key] = float(diff[mask].mean())
    return out

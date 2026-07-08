"""Fit palette-calibration data from a cylindrical reference image.

The calibration algorithm proper: given a decoded RGB reference image (an
``(H, W, 3)`` float array in [0, 1], assumed equirectangular with row 0 at
+90 deg latitude), return palette-row data as PLAIN dicts/arrays. The
per-latitude statistics come from :mod:`gasgiant.palette.reference`.

This module lives in the ``palette`` layer (the lowest layer, below ``gl``)
so it must NOT import ``export``/``render``/``app``/``gl``. In particular the
image DECODE happens in the caller (the CLI/GUI top layer, via
``gasgiant.export.writers.decode_image(path, color=True)``) and the decoded
array is passed in here; the model conversion (plain dicts -> ``PaletteRow``)
likewise happens in the caller, keeping this layer free of ``params`` too.

Fitting happens in display sRGB (pre-AgX); the Blender Cycles render remains
the saturation gate. See ``scripts/calibrate_palette.py`` for the CLI wrapper
and the ``--fit-mode``/``--stops``/``--min-l-span`` semantics.
"""

from __future__ import annotations

import numpy as np

from gasgiant.palette.reference import anchor_fit, expand_stop_span, latitude_profile

# Default anchor latitudes (signed degrees, north positive) and the half-width
# of the latitude window sampled around each anchor.
DEFAULT_ANCHORS: tuple[float, ...] = (-65.0, -40.0, -15.0, 10.0, 40.0, 65.0)
DEFAULT_WINDOW_DEG = 9.0


def _rgb(values: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in values]


def calibrate(
    img: np.ndarray,
    anchors: tuple[float, ...] = DEFAULT_ANCHORS,
    bins: int = 90,
    window_deg: float = DEFAULT_WINDOW_DEG,
    fit_mode: str = "median",
    chroma_pct: float = 0.6,
    stops: int = 3,
    min_l_span: float = 0.0,
) -> dict:
    """Fit palette-calibration data from a decoded RGB reference image.

    ``img`` is an ``(H, W, 3)`` float array in [0, 1] (decoded by the caller).
    Returns a plain-``dict`` document with ``palette_rows`` (each a
    ``{"latitude": deg, "stops": [{"pos": p, "color": [r, g, b]}, ...]}``
    dict), ``contrast_envelope`` and ``latitude_table`` — NO model instances,
    so this stays inside the ``palette`` layer.
    """
    profile = latitude_profile(img, bins)

    rows = []
    for anchor in sorted(anchors):
        if fit_mode == "median" and stops == 3:
            # Original profile-aggregate fit, byte-stable for default flags.
            sel = np.abs(profile.lat_deg - anchor) <= window_deg
            if not sel.any():
                sel = np.argsort(np.abs(profile.lat_deg - anchor))[:3]
            fitted = [
                (0.0, np.median(profile.belt_rgb[sel], axis=0)),
                (0.5, np.median(profile.median_rgb[sel], axis=0)),
                (1.0, np.median(profile.zone_rgb[sel], axis=0)),
            ]
        else:
            # Pixel-level fit (chroma-restore needs member pixels, which the
            # profile aggregates cannot provide; 5-stop uses it too).
            fitted = anchor_fit(img, anchor, window_deg, fit_mode, chroma_pct, stops)
        if min_l_span > 0.0:
            fitted = expand_stop_span(fitted, min_l_span)
        rows.append(
            {
                "latitude": anchor,
                "stops": [{"pos": pos, "color": _rgb(color)} for pos, color in fitted],
            }
        )

    return {
        "palette_rows": rows,
        "contrast_envelope": [
            {"latitude": round(float(lat), 2), "contrast": round(float(c), 4)}
            for lat, c in zip(profile.lat_deg, profile.contrast, strict=True)
        ],
        "latitude_table": [
            {
                "latitude": round(float(profile.lat_deg[i]), 2),
                "zone": _rgb(profile.zone_rgb[i]),
                "belt": _rgb(profile.belt_rgb[i]),
                "median": _rgb(profile.median_rgb[i]),
            }
            for i in range(len(profile.lat_deg))
        ],
    }

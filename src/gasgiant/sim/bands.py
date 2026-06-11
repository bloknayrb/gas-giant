"""Seeded band layout: edge latitudes and per-band color-index/height levels.

Bands are drawn in sin(latitude) space (roughly equal-area) with jittered
widths; values alternate zone-bright / belt-dark with per-band variation.
This layout later also anchors the zonal jet profile (jet maxima sit at band
boundaries) and the relaxation forcing stamps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gasgiant.params.model import BandsParams
from gasgiant.params.seeds import subseed

# Bands cover |lat| <= ~76 deg; poleward caps get dedicated treatment (Phase 3b).
_SIN_LAT_EXTENT = 0.97

ZONE_VALUE = 0.78
BELT_VALUE = 0.30
ZONE_HEIGHT = 0.75
BELT_HEIGHT = 0.35


@dataclass(frozen=True)
class BandLayout:
    edges: np.ndarray  # (count + 1,) latitudes in radians, descending from +pi/2
    values: np.ndarray  # (count,) color index 0..1
    heights: np.ndarray  # (count,) cloud-top height 0..1
    # Faded-sector geometry (SEB fade): lat_lo, lat_hi, center lon, half-width,
    # all radians. Drawn for every layout; applied only when
    # bands.faded_sector > 0.
    fade_sector: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def generate_bands(seed: int, params: BandsParams) -> BandLayout:
    rng = subseed(seed, "bands")
    count = params.count

    widths = 1.0 + params.width_jitter * rng.uniform(-1.0, 1.0, size=count)
    widths = np.maximum(widths, 0.15)
    fractions = np.concatenate([[0.0], np.cumsum(widths) / widths.sum()])
    # Interior edges in sin-lat space, descending; outer edges pinned to the poles
    # so the lookup is defined everywhere (cap styling replaces this in Phase 3b).
    sin_edges = _SIN_LAT_EXTENT - fractions * (2.0 * _SIN_LAT_EXTENT)
    edges = np.arcsin(sin_edges)
    edges[0] = np.pi / 2.0
    edges[-1] = -np.pi / 2.0

    zone_first = bool(rng.integers(0, 2))
    parity = np.arange(count) % 2 == (0 if zone_first else 1)
    base_value = np.where(parity, ZONE_VALUE, BELT_VALUE).astype(np.float64)
    base_height = np.where(parity, ZONE_HEIGHT, BELT_HEIGHT).astype(np.float64)

    mid = 0.5 * (ZONE_VALUE + BELT_VALUE)
    values = mid + (base_value - mid) * params.value_contrast
    values += rng.uniform(-0.06, 0.06, size=count)
    heights = base_height + rng.uniform(-0.08, 0.08, size=count)

    # Per-band palette offset on its own stream: with hue_jitter == 0 the
    # layout is bit-identical to layouts generated before the feature existed.
    hue_rng = subseed(seed, "band-hues")
    values += params.hue_jitter * hue_rng.uniform(-1.0, 1.0, size=count)

    values = np.clip(values, 0.0, 1.0)
    return BandLayout(
        edges=edges.astype(np.float32),
        values=values.astype(np.float32),
        heights=np.clip(heights, 0.0, 1.0).astype(np.float32),
        fade_sector=_select_fade_sector(seed, edges, values),
    )


def _select_fade_sector(
    seed: int, edges: np.ndarray, values: np.ndarray
) -> tuple[float, float, float, float]:
    """The widest low/mid-latitude belt gets the (potential) faded sector;
    longitude and width come from a dedicated stream so drawing them never
    perturbs the band layout."""
    rng = subseed(seed, "faded-sector")
    lon = float(rng.uniform(-np.pi, np.pi))
    halfwidth = float(np.deg2rad(rng.uniform(38.0, 58.0)))

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = -np.diff(edges)
    is_belt = values < np.median(values)
    candidates = np.where(is_belt & (np.abs(centers) < 0.9))[0]
    if candidates.size == 0:
        return (0.0, 0.0, lon, halfwidth)
    j = candidates[np.argmax(widths[candidates])]
    return (float(edges[j + 1]), float(edges[j]), lon, halfwidth)

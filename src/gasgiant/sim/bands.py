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

    return BandLayout(
        edges=edges.astype(np.float32),
        values=np.clip(values, 0.0, 1.0).astype(np.float32),
        heights=np.clip(heights, 0.0, 1.0).astype(np.float32),
    )

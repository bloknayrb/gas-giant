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
    values: np.ndarray  # (count,) color index 0..1 -- PRE-fade (identity/dynamics view)
    heights: np.ndarray  # (count,) cloud-top height 0..1
    # Faded-sector geometry (SEB fade): lat_lo, lat_hi, center lon, half-width,
    # all radians. Drawn for every layout; applied only when
    # bands.faded_sector > 0.
    fade_sector: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Explicit zone/belt identity, (count,) bool, computed ONCE at layout
    # build from `values < median(values)` -- always the PRE-fade values.
    # Consumers (stamp profiles, fade-sector pick, storm seeding, outbreak
    # candidates) must read this instead of re-deriving from values, so a
    # later value edit (belt fade) can never flip a band's class. None ->
    # derived in __post_init__ for hand-built layouts (tests).
    is_belt: np.ndarray | None = None
    # (count,) color values the T0 STAMP path reads (bands.belt_fade applied);
    # `values` above stays pre-fade so the darkest-half outbreak ordering,
    # storm seeding, and identity never move with the fade. None -> aliases
    # `values` in __post_init__ (belt_fade off or hand-built layouts).
    stamp_values: np.ndarray | None = None
    # Band index targeted by belt_fade / faded_sector (the widest-belt pick
    # or the explicit bands.faded_band_index override); None when the layout
    # has no candidate belt.
    fade_index: int | None = None

    def __post_init__(self) -> None:
        if self.is_belt is None:
            values = np.asarray(self.values, dtype=np.float64)
            object.__setattr__(self, "is_belt", values < np.median(values))
        if self.stamp_values is None:
            object.__setattr__(self, "stamp_values", self.values)


def generate_bands(seed: int, params: BandsParams) -> BandLayout:
    if params.template is not None:
        return _bands_from_template(seed, params)
    rng = subseed(seed, "bands")
    count = params.count

    widths = 1.0 + params.width_jitter * rng.uniform(-1.0, 1.0, size=count)
    widths = np.maximum(widths, 0.15)
    # Heavier-tailed width distribution on its own stream (real maps mix very
    # broad zones with thin strips); width_tail == 0 leaves widths untouched.
    tail_rng = subseed(seed, "width-tail")
    widths = widths * np.exp(params.width_tail * tail_rng.normal(0.0, 0.9, size=count))
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
    is_belt = values < np.median(values)
    return _finish_layout(seed, edges, values, np.clip(heights, 0.0, 1.0), is_belt, params)


def _bands_from_template(seed: int, params: BandsParams) -> BandLayout:
    """The explicit-skeleton path: edges/values/heights verbatim (validated
    by BandTemplate -- identity alternation, descending edges, extents).
    NO value seasoning runs here: zone/belt identity is frozen from the
    verbatim values (`values < median(values)`) into BandLayout.is_belt --
    jitter on both a value and the median could flip a band's class.
    Fade-sector selection still applies (its own seed stream; works off any
    edges/values); warp, edge softness/diversity, and detail noise are
    applied downstream of the layout and are unaffected."""
    template = params.template
    edges = np.deg2rad(np.asarray(template.edges_deg, dtype=np.float64))
    values = np.asarray(template.values, dtype=np.float64)
    is_belt = values < np.median(values)
    return _finish_layout(seed, edges, values, np.asarray(template.heights), is_belt, params)


def _finish_layout(
    seed: int,
    edges: np.ndarray,
    values: np.ndarray,
    heights: np.ndarray,
    is_belt: np.ndarray,
    params: BandsParams,
) -> BandLayout:
    """Shared tail of both layout paths: fade-target resolution (heuristic or
    the bands.faded_band_index override), the whole-belt fade on the STAMP
    view of the values, and dtype freezing. The fade runs after every seeded
    draw and touches only stamp_values -- `values`/`is_belt` stay pre-fade."""
    fade_sector, fade_index = _select_fade_sector(
        seed, edges, is_belt, params.faded_band_index
    )
    values32 = values.astype(np.float32)
    return BandLayout(
        edges=edges.astype(np.float32),
        values=values32,
        heights=heights.astype(np.float32),
        fade_sector=fade_sector,
        is_belt=is_belt,
        stamp_values=_apply_belt_fade(values32, fade_index, params.belt_fade),
        fade_index=fade_index,
    )


def _apply_belt_fade(
    values: np.ndarray, fade_index: int | None, amount: float
) -> np.ndarray:
    """STAMP-view values with the whole-belt fade applied: the target band
    blends toward the mean of its neighboring bands (for a belt that is the
    zone level -- the SEB-fade ghost band; for a zone, the belt tans -- the
    ochre-EZ recipe). amount == 0 returns `values` itself so the stamp path
    is byte-identical by construction."""
    if amount <= 0.0 or fade_index is None:
        return values
    neighbors = [
        float(values[k])
        for k in (fade_index - 1, fade_index + 1)
        if 0 <= k < len(values)
    ]
    if not neighbors:
        return values
    target = float(np.mean(neighbors))
    faded = values.copy()
    faded[fade_index] = np.float32(
        float(values[fade_index]) + amount * (target - float(values[fade_index]))
    )
    return faded


def _select_fade_sector(
    seed: int,
    edges: np.ndarray,
    is_belt: np.ndarray,
    override_index: int | None = None,
) -> tuple[tuple[float, float, float, float], int | None]:
    """The widest low/mid-latitude belt gets the (potential) faded sector --
    or the explicit ``bands.faded_band_index`` when set (review B5-4: the
    shipped Jupiter template's widest-belt argmax is a 0.01-degree tiebreak,
    so template tweaks silently moved the fade). Longitude and width come
    from a dedicated stream so drawing them never perturbs the band layout;
    the draws happen BEFORE the pick so an override never shifts them.
    Identity comes in precomputed (see BandLayout.is_belt) -- never re-derive
    it from values here. Returns (sector, band index)."""
    rng = subseed(seed, "faded-sector")
    lon = float(rng.uniform(-np.pi, np.pi))
    halfwidth = float(np.deg2rad(rng.uniform(38.0, 58.0)))

    if override_index is not None:
        j = int(override_index)
        return (float(edges[j + 1]), float(edges[j]), lon, halfwidth), j

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = -np.diff(edges)
    candidates = np.where(is_belt & (np.abs(centers) < 0.9))[0]
    if candidates.size == 0:
        return (0.0, 0.0, lon, halfwidth), None
    j = int(candidates[np.argmax(widths[candidates])])
    return (float(edges[j + 1]), float(edges[j]), lon, halfwidth), j

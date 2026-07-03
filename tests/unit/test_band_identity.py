"""Explicit zone/belt identity on BandLayout (2026-07-02 review, D2 foundation).

Identity is frozen once at layout build; the stamp profiles, fade-sector
pick, storm seeding, and outbreak candidates all read BandLayout.is_belt
instead of re-deriving `values < median(values)`, so a later value edit
(belt fade, W6) can never flip a band's class.
"""

from __future__ import annotations

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.events import EventSchedule
from gasgiant.sim.vortices import MAX_VORTEX_LAT, _band_centers

FACTORY = ["gas_giant_warm", "jupiter_vorticity", "jupiter_like", "saturn_pale", "ice_giant"]

def test_bandlayout_is_belt_matches_legacy_formula_everywhere():
    """The explicit identity must equal the `values < median(values)` mask
    every consumer used to re-derive, for all factory presets and a spread
    of procedural seeds."""
    layouts = []
    for name in FACTORY:
        p = load_factory_preset(name)
        layouts.append(generate_bands(p.seed, p.bands))
    for seed in (1, 5, 4201, 999):
        layouts.append(generate_bands(seed, PlanetParams(seed=seed).bands))
    for bands in layouts:
        legacy = bands.values < np.median(bands.values)
        assert bands.is_belt.dtype == np.bool_
        np.testing.assert_array_equal(bands.is_belt, legacy)


def test_band_centers_and_events_use_explicit_identity():
    """_band_centers and outbreak candidate selection agree with is_belt
    (guards the four-consumer refactor: profiles, fade-sector, vortices,
    events all read the same mask)."""
    p = load_factory_preset("jupiter_vorticity")
    bands = generate_bands(p.seed, p.bands)
    centers_belt = {c for c, _ in _band_centers(bands, want_belt=True)}
    centers_zone = {c for c, _ in _band_centers(bands, want_belt=False)}
    for j in range(len(bands.values)):
        center = float(0.5 * (bands.edges[j] + bands.edges[j + 1]))
        if abs(center) > MAX_VORTEX_LAT:
            continue
        expected = centers_belt if bands.is_belt[j] else centers_zone
        other = centers_zone if bands.is_belt[j] else centers_belt
        assert center in expected
        assert center not in other

    sched = EventSchedule.generate(p, bands)
    belt_lats = {float(0.5 * (bands.edges[j] + bands.edges[j + 1]))
                 for j in range(len(bands.values)) if bands.is_belt[j]}
    for ob in sched.outbreaks:
        assert any(abs(ob.lat - bl) < 0.06 for bl in belt_lats)

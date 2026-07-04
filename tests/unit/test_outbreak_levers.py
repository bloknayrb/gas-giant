"""Outbreak placement levers (2026-07-02 review B5-3/B5-8, W6).

storms.outbreak_latitude (degrees, nullable) pins WHERE convective outbreaks
erupt -- the Great-White-Spot-at-35N unlock, and (because the pin bypasses
the belt-candidate machinery entirely) the 1990 equatorial-class GWS too.
storms.outbreak_phase (0..1, nullable) pins WHEN: the eruption start step as
a fraction of the dev run, replacing the seeded 0.55..0.85 draw.
storms.outbreak_lat_min promotes the previously hardcoded 0.20 rad candidate
floor so AUTO selection can admit equatorial belts without pinning.

All three default to current behavior; the seeded draw SEQUENCE is preserved
when a pin is active (the draw is consumed, then overridden), so toggling a
pin never reshuffles the other outbreak properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.params.model import PlanetParams, StormsParams
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import BandLayout, generate_bands
from gasgiant.sim.events import TRAIN_LAT_SPREAD, TRAIN_N, EventSchedule


def _planet(count: int = 2, dev_steps: int = 500, **storms) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p.sim.dev_steps = dev_steps
    p.storms.outbreak_count = count
    for key, value in storms.items():
        setattr(p.storms, key, value)
    return p


def _schedule(**storms) -> EventSchedule:
    p = _planet(**storms)
    bands = generate_bands(p.seed, p.bands)
    return EventSchedule.generate(p, bands)


# ------------------------------------------------------------------ defaults

def test_forced_variant_noop_explicit_defaults():
    base = _schedule()
    forced = _schedule(outbreak_latitude=None, outbreak_phase=None,
                       outbreak_lat_min=0.20)
    assert forced == base
    assert base.outbreaks  # sanity: the preset actually schedules eruptions


# ------------------------------------------------------------- latitude pin

def test_outbreak_latitude_pins_all_eruptions():
    """~35N GWS case: every knot sits within the train's tight latitude
    bracket around the pinned center."""
    sched = _schedule(outbreak_latitude=35.0)
    assert sched.outbreaks
    center = np.deg2rad(35.0)
    for ob in sched.outbreaks:
        assert abs(ob.lat - center) <= 0.5 * TRAIN_LAT_SPREAD + 1e-9


def test_outbreak_latitude_reaches_the_equator():
    """The 1990 equatorial-class GWS: a pin at 0 works even though the auto
    candidate floor excludes equatorial belts."""
    sched = _schedule(outbreak_latitude=0.0)
    assert sched.outbreaks
    for ob in sched.outbreaks:
        assert abs(ob.lat) <= 0.5 * TRAIN_LAT_SPREAD + 1e-9


def test_outbreak_latitude_preserves_other_draws():
    """Pinning consumes the same seeded draws, so longitudes and steps match
    the unpinned schedule knot for knot."""
    base = _schedule()
    pinned = _schedule(outbreak_latitude=35.0)
    assert len(pinned.outbreaks) == len(base.outbreaks)
    for a, b in zip(base.outbreaks, pinned.outbreaks, strict=True):
        assert a.lon == b.lon
        assert a.step == b.step
        assert a.radius == b.radius


# ---------------------------------------------------------------- phase pin

def test_outbreak_phase_pins_eruption_start():
    dev_steps = 500
    sched = _schedule(outbreak_phase=0.3, dev_steps=dev_steps)
    assert sched.outbreaks
    start = int(0.3 * dev_steps)
    # Lead knots (largest radius per train) start exactly at the pinned step
    # plus the per-knot seeded stagger window; every knot lands in
    # [start, start + train stagger + jitter window].
    max_stagger = (TRAIN_N - 1) * int(0.015 * dev_steps) + int(0.04 * dev_steps)
    for ob in sched.outbreaks:
        assert start <= ob.step <= start + max_stagger
    assert min(ob.step for ob in sched.outbreaks) < start + int(0.04 * dev_steps) + 1


def test_outbreak_phase_preserves_placement_draws():
    base = _schedule()
    phased = _schedule(outbreak_phase=0.6)
    for a, b in zip(base.outbreaks, phased.outbreaks, strict=True):
        assert a.lat == b.lat
        assert a.lon == b.lon


# ---------------------------------------------------------- candidate floor

def _equatorial_layout() -> BandLayout:
    """Hand-built layout with one belt centered at 0.05 rad (inside the
    default 0.20 rad floor) and nothing else in the candidate window."""
    edges = np.array([np.pi / 2, 0.2, -0.1, -np.pi / 2], dtype=np.float32)
    values = np.array([0.8, 0.3, 0.78], dtype=np.float32)
    heights = np.array([0.7, 0.4, 0.7], dtype=np.float32)
    return BandLayout(edges=edges, values=values, heights=heights)


def test_outbreak_lat_min_default_excludes_equatorial_belt():
    p = _planet()
    sched = EventSchedule.generate(p, _equatorial_layout())
    assert sched.outbreaks == []


def test_outbreak_lat_min_zero_admits_equatorial_belt():
    p = _planet(outbreak_lat_min=0.0)
    sched = EventSchedule.generate(p, _equatorial_layout())
    assert sched.outbreaks
    for ob in sched.outbreaks:
        assert abs(ob.lat - 0.05) <= 0.5 * TRAIN_LAT_SPREAD + 1e-6


def test_outbreak_latitude_pin_works_without_candidates():
    """A pin bypasses the belt-candidate machinery: it erupts even when no
    belt clears the auto-selection window."""
    p = _planet(outbreak_latitude=10.0)
    sched = EventSchedule.generate(p, _equatorial_layout())
    assert sched.outbreaks
    center = np.deg2rad(10.0)
    for ob in sched.outbreaks:
        assert abs(ob.lat - center) <= 0.5 * TRAIN_LAT_SPREAD + 1e-9


# ------------------------------------------------------------------ validators

def test_outbreak_latitude_bounds():
    with pytest.raises(ValueError):
        StormsParams(outbreak_latitude=70.0)
    with pytest.raises(ValueError):
        StormsParams(outbreak_latitude=-70.0)
    StormsParams(outbreak_latitude=35.0)


def test_outbreak_phase_bounds():
    with pytest.raises(ValueError):
        StormsParams(outbreak_phase=1.5)
    with pytest.raises(ValueError):
        StormsParams(outbreak_phase=-0.1)
    StormsParams(outbreak_phase=0.0)
    StormsParams(outbreak_phase=1.0)

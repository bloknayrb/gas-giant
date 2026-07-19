"""Facade seat-meter exposure: the GUI reads the natural-bearing quality through
Simulation, computed on the BRACKET-OFF profile (so it reports the natural
bearing even when the override is on).

WHITE-BOX construction (deliberate): Simulation.__init__ has NO CPU-only path --
it calls GpuContext.headless() (needs GL 4.3) and _build() (allocates GPU LUTs),
and there is zero Simulation() usage in tests/unit. But seat_quality/seat_status
read ONLY self.params and self.bands, both CPU-constructible. So we bypass
__init__ with object.__new__ and inject exactly those two attributes, keeping
these behavioral tests in the no-GPU tier. A future edit that makes the seat
methods touch self.solver/self.gpu will surface here as an AttributeError --
that is the intended tripwire."""
from __future__ import annotations

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.sim.bands import generate_bands


def _seat_sim(params: PlanetParams) -> Simulation:
    sim = object.__new__(Simulation)          # bypass GPU __init__
    sim.params = params
    sim.bands = generate_bands(params.seed, params.bands)
    sim._seat_profile = None                  # the lazy bracket-off cache slot
    return sim


def test_seat_quality_none_without_pinned_hero():
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = None
    sim = _seat_sim(p)
    assert sim.seat_quality() is None
    assert sim.seat_status() is None


def test_seat_quality_uses_bracket_off_profile():
    """With the bracket ON, seat_quality still reports the NATURAL bearing (it
    builds the profile with hero_lat_deg=None, which skips the override), so
    turning the bracket on does not change the meter reading."""
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = -22.0
    off = _seat_sim(p).seat_quality()
    p2 = PlanetParams(seed=4201)
    p2.storms.hero_latitude = -22.0
    p2.jets.hero_bracket_north = -1.0
    p2.jets.hero_bracket_south = 0.6
    on = _seat_sim(p2).seat_quality()
    assert on == off


def test_seat_quality_reads_draft_latitude_override():
    """The GUI passes the live (draft) hero latitude; seat_quality(lat_deg=...)
    samples there without rebuilding, so a scan is cheap."""
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = -22.0
    sim = _seat_sim(p)
    assert sim.seat_quality(-40.0) != sim.seat_quality(-22.0)


def test_seat_status_is_a_banded_string():
    p = PlanetParams(seed=4201)
    p.storms.hero_latitude = -22.0
    s = _seat_sim(p).seat_status()
    assert any(band in s for band in ("green", "amber", "red"))

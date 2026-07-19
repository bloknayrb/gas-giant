"""_hero_r_core: the hero core radius threaded into build_profiles for the
size-relative bracket geometry. Pure function -- no GL, no-GPU tier."""
from __future__ import annotations

from gasgiant.engine.facade import _hero_r_core
from gasgiant.params.model import PlanetParams


def test_hero_r_core_returns_the_hero_radius():
    p = PlanetParams(seed=1)
    p.storms.hero_radius = 0.077
    assert _hero_r_core(p) == 0.077


def test_hero_r_core_is_the_radius_regardless_of_pin():
    """Not None-able: it returns the radius even with no pinned hero, because
    build_profiles ignores it when the override block is skipped."""
    p = PlanetParams(seed=1)          # hero_latitude None by default
    assert _hero_r_core(p) == p.storms.hero_radius

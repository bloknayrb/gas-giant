"""GPU: hero anatomy (collar/perimeter) and the bright wake reach tracers."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _hero_sim(gpu) -> Simulation:
    p = PlanetParams(seed=33)
    p.sim.resolution = 1024
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.waves.festoon_strength = 0.0
    return Simulation(p, gpu)


def test_wake_is_bright_and_cool_downstream(gpu):
    sim = _hero_sim(gpu)
    hero = sim.vortices.heroes()[0]
    t = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)
    h, w = t.shape[:2]

    def sample(dlon_radians: float):
        lon = (hero.lon + dlon_radians + np.pi) % (2 * np.pi) - np.pi
        x = int((lon + np.pi) / (2 * np.pi) * w) % w
        y = int((np.pi / 2 - hero.lat) / np.pi * h)
        region = t[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3]
        return region[..., 0].mean(), region[..., 3].mean()

    wake_t0, wake_t3 = sample(hero.wake_dir * hero.r_core * 3.0)
    quiet_t0, quiet_t3 = sample(-hero.wake_dir * hero.r_core * 6.0)
    assert wake_t0 > quiet_t0 + 0.04   # bright churned clouds downstream
    assert wake_t3 < quiet_t3 - 0.05   # cool gray-white tint


def test_hero_has_dark_perimeter_and_bright_collar(gpu):
    sim = _hero_sim(gpu)
    hero = sim.vortices.heroes()[0]
    t = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)
    h, w = t.shape[:2]

    def t0_at(q: float):
        # Sample on the upstream side, away from the wake wedge.
        lon = (hero.lon - hero.wake_dir * q * hero.r_core + np.pi) % (2 * np.pi) - np.pi
        x = int((lon + np.pi) / (2 * np.pi) * w) % w
        y = int((np.pi / 2 - hero.lat) / np.pi * h)
        return float(t[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2, 0].mean())

    perimeter = t0_at(1.0)
    collar = t0_at(1.55)
    far = t0_at(4.0)
    # The T0 clamp at 1.0 compresses the bright side, so the separation is
    # smaller than the raw stamp amplitudes suggest.
    assert collar > perimeter + 0.05  # bright hollow outside the dark ring
    assert collar > far + 0.04

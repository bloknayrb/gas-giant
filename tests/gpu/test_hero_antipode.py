"""GPU regression: an elongated hero must NOT stamp a duplicate at its antipode.

The aspect-elongation path projects sphere points onto the hero's east/north
tangent frame: q = |(p.e1/asp, p.e2)| / r. At the ANTIPODE (p = -center) both
tangent components vanish, so q -> 0 and the full hero anatomy (tint collar +
mottle, and the vorticity patch) was stamped a second time on the far side of the
planet -- a phantom "second Great Red Spot". The round path (q = d/r) is immune
because d -> pi at the antipode. Fix: gate the elliptical metric to the near
hemisphere (dot(p, center) > 0).

This is a correctness fix, NOT default-off: it changes any preset with
hero_aspect != 1.0 (removes the phantom). aspect == 1.0 stays byte-identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

HERO_LAT_DEG = -22.5


def _sim(aspect: float):
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    p.storms.hero_count = 1
    p.storms.hero_latitude = HERO_LAT_DEG
    p.storms.hero_aspect = aspect
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.small_density = 0.0
    return Simulation(p, gpu_ctx)


def _box_t3(tr: np.ndarray, lat_deg: float, lon_deg: float, half: int = 14) -> float:
    h, w = tr.shape[:2]
    r = int((90.0 - lat_deg) / 180.0 * h)
    c = int(((lon_deg + 180.0) % 360.0) / 360.0 * w)
    rows = slice(max(r - half, 0), min(r + half, h))
    # longitude wrap: roll so the column is centered, then crop
    rolled = np.roll(tr[..., 3], w // 2 - c, axis=1)
    return rolled[rows, w // 2 - half:w // 2 + half].max()


@pytest.fixture(autouse=True)
def _ctx(gpu):
    global gpu_ctx
    gpu_ctx = gpu
    yield


def test_elongated_hero_has_no_antipode_stamp(gpu):
    sim = _sim(aspect=2.0)
    tr = gpu.read_texture(sim.solver.equirect.tracers.cur)
    hero = sim.vortices.heroes()[0]
    hlat, hlon = np.degrees(hero.lat), np.degrees(hero.lon)

    near = _box_t3(tr, hlat, hlon)                       # the real hero
    anti = _box_t3(tr, -hlat, hlon + 180.0)              # its antipode

    assert near > 0.5, f"real hero stamp missing (near T3={near:.3f})"
    assert anti < 0.3, (
        f"phantom hero at antipode: T3={anti:.3f} (should be ~background, the "
        f"elliptical q aliases at the antipode)"
    )


def test_round_hero_antipode_also_clean(gpu):
    """Sanity: a round hero (aspect 1.0) never had the bug; confirm it's clean."""
    sim = _sim(aspect=1.0)
    tr = gpu.read_texture(sim.solver.equirect.tracers.cur)
    hero = sim.vortices.heroes()[0]
    hlat, hlon = np.degrees(hero.lat), np.degrees(hero.lon)
    assert _box_t3(tr, -hlat, hlon + 180.0) < 0.3

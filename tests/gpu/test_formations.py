from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.sim.events import LIFETIME, TRAIN_N, EventSchedule

pytestmark = pytest.mark.gpu


def _params(seed: int = 21, steps: int = 0) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    return p


def test_festoons_stamp_negative_tint(gpu):
    p = _params()
    p.waves.festoon_strength = 1.2
    sim = Simulation(p, gpu)
    sim.solver.step(80)
    t3 = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 3]
    assert t3.min() < -0.15, f"no festoon (negative T3) signature: min {t3.min():.3f}"

    off = _params()
    off.waves.festoon_strength = 0.0
    sim2 = Simulation(off, gpu)
    sim2.solver.step(80)
    t3_off = sim2.gpu.read_texture(sim2.solver.equirect.tracers.cur)[..., 3]
    assert t3.min() < t3_off.min() - 0.1


def test_outbreak_spawns_and_retires(gpu):
    p = _params(steps=0)
    p.sim.dev_steps = 400
    p.storms.outbreak_count = 1
    sim = Simulation(p, gpu)
    schedule: EventSchedule = sim.solver.events
    # One eruption_count spawns a TRAIN of knots (a belt-girdling chain), not a
    # single spot — see events.py generate().
    assert len(schedule.outbreaks) == TRAIN_N
    baseline = len(sim.vortices.vortices)
    first = min(ob.step for ob in schedule.outbreaks)
    last = max(ob.step for ob in schedule.outbreaks)

    # Step to just after the first knot erupts: at least one train vortex is live.
    sim.solver.step(first + 10 - sim.solver.step_index)
    assert len(sim.vortices.vortices) > baseline, "no outbreak vortex spawned"

    # Step well past the last knot's lifetime: the whole train has retired.
    sim.solver.step((last + LIFETIME + 10) - sim.solver.step_index)
    assert len(sim.vortices.vortices) == baseline, "outbreak train not retired"


def test_outbreak_brightens_the_region(gpu):
    p = _params(seed=33)
    p.sim.dev_steps = 400  # schedule needs a real run window; we step manually
    p.storms.outbreak_count = 1
    p.storms.outbreak_strength = 1.5
    sim = Simulation(p, gpu)
    ob = sim.solver.events.outbreaks[0]
    # Snapshot just after eruption peak.
    sim.solver.step(ob.step + 30)
    t0 = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]
    h, w = t0.shape
    cy = int((np.pi / 2 - ob.lat) / np.pi * h)
    cx = int((ob.vortex.lon + np.pi) / (2 * np.pi) * w) if ob.vortex else int(
        (ob.lon + np.pi) / (2 * np.pi) * w
    )
    region = t0[max(cy - 12, 0) : cy + 12, max(cx - 12, 0) : cx + 12]
    surround = t0[max(cy - 12, 0) : cy + 12, :]
    assert region.max() > np.percentile(surround, 95), "outbreak plume not brighter than band"


def test_detail_synthesis_adds_structure(gpu):
    p = _params(steps=40)
    sim = Simulation(p, gpu)
    sim.run_to_completion()

    sim.params.detail.intensity = 0.0
    sim._post_dirty = True
    flat = sim.gpu.read_texture(sim.ensure_preview(512)[0])

    sim.params.detail.intensity = 1.0
    sim._post_dirty = True
    detailed = sim.gpu.read_texture(sim.ensure_preview(512)[0])

    assert not np.array_equal(flat, detailed)
    # Detail must add high-frequency energy (gradient magnitude increases).
    g_flat = np.abs(np.diff(flat[..., 0], axis=1)).mean()
    g_det = np.abs(np.diff(detailed[..., 0], axis=1)).mean()
    assert g_det > g_flat


def test_ribbon_darkens_a_line(gpu):
    p = _params(seed=5)
    p.waves.ribbon_strength = 1.5
    sim = Simulation(p, gpu)
    sim.solver.step(40)
    t0 = sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]
    off = _params(seed=5)
    off.waves.ribbon_strength = 0.0
    sim2 = Simulation(off, gpu)
    sim2.solver.step(40)
    t0_off = sim2.gpu.read_texture(sim2.solver.equirect.tracers.cur)[..., 0]
    rib_lat = sim.solver.wave_lats[1]
    h = t0.shape[0]
    row = int((np.pi / 2 - rib_lat) / np.pi * h)
    assert t0[row].mean() < t0_off[row].mean() - 0.02

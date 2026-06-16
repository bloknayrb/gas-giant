"""M3 coupling integration: the evolving baroclinic source develops a v1.6 run,
records coverage, changes the render, and stays coherent."""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.baroclinic_coupling import run_coupled
from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver

pytestmark = pytest.mark.gpu


def _params(steps: int) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    return p


def test_coupled_run_develops_and_changes_render(gpu):
    base = Simulation(_params(steps=16), gpu)
    base_rgb = np.clip(base.render_maps(512)["color"][..., :3], 0, 1)
    base._release_sim()

    sim = Simulation(_params(steps=16), gpu)
    w, h = sim.solver.equirect.size
    driver = BaroclinicSourceDriver(grid_w=w, grid_h=h, warmup_steps=3000, seed=0)
    stats = run_coupled(sim, driver, gain=1.0, update_every=4,
                        baro_steps_per_update=100)
    coupled_rgb = np.clip(sim.render_maps(512)["color"][..., :3], 0, 1)
    sim._release_sim()

    assert sim.is_developed
    assert stats.v16_steps >= 16
    assert stats.source_updates >= 4
    assert float(np.abs(coupled_rgb - base_rgb).mean()) > 1e-4

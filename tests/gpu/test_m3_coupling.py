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


def test_coupled_concentrates_eddies_in_band(gpu):
    from gasgiant.render.m3_metrics import highfreq_energy, latitude_concentration

    base = Simulation(_params(steps=24), gpu)
    base_rgb = np.clip(base.render_maps(512)["color"][..., :3], 0, 1)
    base._release_sim()

    sim = Simulation(_params(steps=24), gpu)
    w, h = sim.solver.equirect.size
    driver = BaroclinicSourceDriver(grid_w=w, grid_h=h, warmup_steps=6000, seed=0)
    run_coupled(sim, driver, gain=1.2, update_every=4, baro_steps_per_update=200)
    coupled_rgb = np.clip(sim.render_maps(512)["color"][..., :3], 0, 1)
    sim._release_sim()

    base_conc = latitude_concentration(base_rgb)
    coupled_conc = latitude_concentration(coupled_rgb)
    # The source concentrates eddies in the active band (relative gain).
    assert coupled_conc >= base_conc, (coupled_conc, base_conc)
    # Natural texture preserved, not smoothed away (within 0.5x..2x of baseline).
    ratio = highfreq_energy(coupled_rgb) / (highfreq_energy(base_rgb) + 1e-12)
    assert 0.5 <= ratio <= 2.0, ratio

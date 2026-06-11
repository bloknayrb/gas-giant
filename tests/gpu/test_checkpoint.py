from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.engine.checkpoint import load_checkpoint, save_checkpoint
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def test_checkpoint_round_trip(gpu, tmp_path):
    p = PlanetParams(seed=55)
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    sim = Simulation(p, gpu)
    sim.solver.step(30)
    path = tmp_path / "state.npz"
    save_checkpoint(sim, path)

    restored = load_checkpoint(path, gpu)
    assert restored.steps_done == 30
    np.testing.assert_array_equal(
        restored.tracers.read_current(), sim.tracers.read_current()
    )

    # Continuing from the restored state matches continuing the original.
    sim.solver.step(10)
    restored.solver.step(10)
    np.testing.assert_allclose(
        restored.tracers.read_current(), sim.tracers.read_current(), atol=1e-6
    )

"""Phase 5: Simulation.rebuild() -- the "Restart dev" action's engine hook.

rebuild() is just _release_sim()+_build() (the same pairing update_params
already uses for a RESTART-tier diff) exposed standalone so the UI can re-init
deterministically from the current params without needing a params diff to
trigger it. These tests pin: (1) rebuild() after stepping returns to step 0
and reproduces a byte-identical fresh Simulation built from the same params,
and (2) tick() itself is untouched -- continuing after rebuild() matches
continuing a fresh sim step-for-step.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _quick_params(seed: int = 7) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512  # SimParams.resolution's floor
    p.sim.dev_steps = 40
    return p


def test_rebuild_resets_step_index(gpu):
    p = _quick_params()
    sim = Simulation(p, gpu)
    sim.solver.step(15)
    assert sim.steps_done == 15

    sim.rebuild()
    assert sim.steps_done == 0
    assert sim.is_developed is False


def test_rebuild_matches_fresh_simulation(gpu):
    """A rebuilt sim must be indistinguishable from `Simulation(params)` built
    from scratch -- concrete field comparison (tracers), not just "didn't
    crash". Ticked partway first so rebuild() actually has state to discard."""
    p = _quick_params(seed=11)
    sim = Simulation(p, gpu)
    sim.solver.step(20)
    assert sim.steps_done == 20  # sanity: state actually diverged pre-rebuild

    sim.rebuild()

    fresh = Simulation(p, gpu)
    assert sim.steps_done == fresh.steps_done == 0
    np.testing.assert_array_equal(
        sim.tracers.read_current(), fresh.tracers.read_current()
    )


def test_rebuild_then_tick_matches_fresh_run(gpu):
    """tick() itself is unmodified by rebuild(): stepping a rebuilt sim must
    reproduce the same trajectory as stepping a never-touched fresh sim."""
    p = _quick_params(seed=23)
    sim = Simulation(p, gpu)
    sim.solver.step(30)
    sim.rebuild()

    fresh = Simulation(p, gpu)

    while sim.tick(4):
        fresh.tick(4)
    assert not fresh.tick(4)  # both should finish at the same target

    assert sim.steps_done == fresh.steps_done
    np.testing.assert_allclose(
        sim.tracers.read_current(), fresh.tracers.read_current(), atol=1e-6
    )

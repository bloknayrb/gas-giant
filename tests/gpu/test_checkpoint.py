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


def test_checkpoint_from_older_generation_is_refused(gpu, tmp_path):
    """A checkpoint pairs saved tracers with a registry replayed from the
    seed; if the generation algorithms changed since, it must refuse loudly
    rather than load and render subtly wrong."""
    import json

    from gasgiant.engine.checkpoint import GENERATION_VERSION
    from gasgiant.params.presets import to_preset_doc

    p = PlanetParams(seed=5)
    p.sim.resolution = 512
    path = tmp_path / "old.npz"
    blank = np.zeros((1, 1, 4), dtype=np.float32)
    # A v1-era checkpoint: no generation_version key at all.
    np.savez_compressed(
        path, preset=json.dumps(to_preset_doc(p)), step_index=3,
        tracers_eq=blank, tracers_n=blank, tracers_s=blank,
    )
    with pytest.raises(ValueError, match="generation_version"):
        load_checkpoint(path, gpu)
    assert GENERATION_VERSION >= 2

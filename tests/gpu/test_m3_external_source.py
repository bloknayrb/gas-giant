"""M3 external vorticity source hook: off-path byte-identity, on-path effect,
leak-free re-upload (the evolving-injection path re-uploads every cadence)."""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu


def _vort_params(seed: int = 7, steps: int = 6) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={"seed": seed})
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    return p


def _equirect_source(sim: Simulation, value: float) -> np.ndarray:
    w, h = sim.solver.equirect.size
    return np.full((h, w), value, dtype=np.float32)


def test_rebind_reuses_texture(gpu):
    """Re-uploading a same-size source must NOT allocate a new texture each call
    (the evolving path calls this thousands of times)."""
    sim = Simulation(_vort_params(steps=0), gpu)
    try:
        sim.set_external_vorticity_source(_equirect_source(sim, 0.1), gain=0.5)
        first = sim.solver.external_omega_tex
        assert first is not None
        sim.set_external_vorticity_source(_equirect_source(sim, 0.2), gain=0.5)
        second = sim.solver.external_omega_tex
        assert second is first, "same-size re-upload must reuse the texture object"
    finally:
        sim._release_sim()


def _render_bytes(sim: Simulation) -> bytes:
    return sim.render_maps(512)["color"].tobytes()


def test_off_path_byte_identical(gpu):
    """gain=0 (with a bound field) renders byte-identically to a never-touched
    default run."""
    base = Simulation(_vort_params(), gpu)
    base_bytes = _render_bytes(base)
    base._release_sim()

    sim = Simulation(_vort_params(), gpu)
    sim.set_external_vorticity_source(_equirect_source(sim, 1.0), gain=0.0)
    off_bytes = _render_bytes(sim)
    sim._release_sim()

    assert off_bytes == base_bytes, "gain==0 must be byte-identical to default"


def test_nonzero_gain_changes_output(gpu):
    """A nonzero gain on a nonzero source must change the render."""
    base = Simulation(_vort_params(), gpu)
    base_bytes = _render_bytes(base)
    base._release_sim()

    sim = Simulation(_vort_params(), gpu)
    sim.set_external_vorticity_source(_equirect_source(sim, 1.0), gain=1.5)
    on_bytes = _render_bytes(sim)
    sim._release_sim()

    assert on_bytes != base_bytes, "nonzero gain must alter the render"

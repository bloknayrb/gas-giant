"""M3 external vorticity source hook: off-path byte-identity, on-path effect,
leak-free re-upload (the evolving-injection path re-uploads every cadence)."""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.solver import DOMAIN_EQUIRECT, DOMAIN_NORTH

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


def _omega_rel(sim: Simulation, domain: int) -> np.ndarray:
    return np.asarray(sim.gpu.read_texture(sim.solver._omega_states[domain].omega_rel))


def test_polar_domains_ignore_external_source(gpu):
    """The injection gate is equirect-only (solver binds the source only when
    dom.kind == DOMAIN_EQUIRECT; all other domains get u_external_gain=0.0). This
    gate was rewritten in the RHS-injection fix (moved from the force pass to the
    recover pass), so pin it: a bound source at nonzero gain must leave the polar
    patches' Poisson RHS byte-identical to an uncoupled run, while the equirect
    band's RHS changes."""
    base = Simulation(_vort_params(), gpu)
    base_rgb = _render_bytes(base)  # develop the sim (run_to_completion)
    base_north = _omega_rel(base, DOMAIN_NORTH)
    base_equi = _omega_rel(base, DOMAIN_EQUIRECT)
    base._release_sim()

    sim = Simulation(_vort_params(), gpu)
    sim.set_external_vorticity_source(_equirect_source(sim, 1.0), gain=1.5)
    _render_bytes(sim)
    sim_north = _omega_rel(sim, DOMAIN_NORTH)
    sim_equi = _omega_rel(sim, DOMAIN_EQUIRECT)
    sim._release_sim()

    # Polar patch RHS must be untouched by the equirect-only source.
    assert np.array_equal(sim_north, base_north), (
        "external source leaked into a polar domain -- the equirect-only gate regressed"
    )
    # Positive control: the equirect band's RHS does see the source.
    assert not np.array_equal(sim_equi, base_equi)
    assert base_rgb is not None

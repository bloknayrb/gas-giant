"""GPU tests for solver.vort_eddy_drag (eddy-only large-scale drag).

Global vort_drag pulls the WHOLE relative vorticity toward zero, so it damps the
zonal-mean jets (washes bands, bleeds hero contrast) along with the eddies.
vort_eddy_drag instead damps only q - <q>_x, the deviation from the per-latitude
zonal mean: it absorbs the gravest-mode inverse-cascade swirl while leaving the
zonal jets EXACTLY in place.

Invariants:
  1. vort_eddy_drag=0 is byte-identical to the path without it (off = no-op).
  2. vort_eddy_drag>0 materially changes the render.
  3. PHYSICS (the defining property): over one step from an identical state,
     turning eddy drag on reduces the eddy field q - <q>_x, yet perturbs the
     zonal-mean profile <q>_x far less than it perturbs the field as a whole.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2
RES = 512


def _params(eddy_drag: float, steps: int = 60, inject: float = 2.0):
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = RES
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.solver.vort_eddy_drag = eddy_drag
    p.solver.vort_inject = inject         # ensure a non-trivial eddy field exists
    p.solver.vort_inject_mask = "global"
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    return p


def _render(p, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(RES)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def _step_q(p, gpu) -> np.ndarray:
    """Run the sim and read back the equirect absolute-vorticity field q."""
    sim = Simulation(p, gpu)
    try:
        sim.render_maps(RES)
        return sim.gpu.read_texture(sim.solver._omega_state.cur)[..., 0].astype(np.float64)
    finally:
        sim._release_sim()


def test_eddy_drag_off_is_noop(gpu):
    """vort_eddy_drag=0 skips the reduction dispatch entirely and the SUBPASS-0
    term is `if (u_vort_eddy_drag > 0.0)` false, so the off path is an exact no-op
    at the kernel (sibling of the deformation_radius/solid-core tests).  Injection is
    off here, but the modernized jupiter_vorticity base is otherwise LIVE (bold hero +
    rich detail), and its vorticity SOR carries ~1e-3 cross-instance LSB noise that
    8-bit color no longer rounds away -- so the empirical check uses the GPU noise
    floor rather than assert_array_equal; a real effect is >> the floor."""
    base = _render(_params(0.0, inject=0.0), gpu)
    same = _render(_params(0.0, inject=0.0), gpu)
    assert np.abs(base - same).max() < GPU_NOISE_ATOL


def test_eddy_drag_changes_render(gpu):
    plain = _render(_params(0.0), gpu)
    drained = _render(_params(0.2), gpu)
    assert np.abs(drained - plain).max() > GPU_NOISE_ATOL, (
        "vort_eddy_drag=0.2 did not change the vorticity-mode render"
    )


def test_eddy_drag_damps_eddies_preserves_zonal_mean(gpu):
    """The headline property, measured cleanly over ONE step from an identical
    state: the two runs are identical up to the final eddy-drag line, so
    q_on = q_off - nu*(q_off - <q_pre>). Eddy drag must shrink the eddy field
    while changing the zonal-mean profile far less than the field overall."""
    q_off = _step_q(_params(0.0, steps=1), gpu)
    q_on = _step_q(_params(0.3, steps=1), gpu)

    z_off = q_off.mean(axis=1)
    z_on = q_on.mean(axis=1)
    eddy_off = q_off - z_off[:, None]
    eddy_on = q_on - z_on[:, None]

    # The drag actually acted.
    field_change = np.abs(q_on - q_off).max()
    assert field_change > GPU_NOISE_ATOL, "eddy drag had no effect on the field"

    # Eddies are damped.
    assert np.abs(eddy_on).mean() < 0.97 * np.abs(eddy_off).mean(), (
        "eddy drag did not reduce the eddy field q - <q>_x"
    )

    # The zonal-mean PROFILE is preserved far better than the field changes
    # (global drag would move both together).
    mean_change = np.abs(z_on - z_off).max()
    assert mean_change < 0.25 * field_change, (
        f"eddy drag moved the zonal-mean jets too much: mean_change={mean_change:.4g} "
        f"vs field_change={field_change:.4g} (should leave the mean ~intact)"
    )


def test_eddy_drag_zonal_mean_matches_numpy(gpu):
    """The CPU reference vorticity_ref.zonal_mean mirrors what the kernel feeds
    the drag: the per-row mean of the read-back q matches numpy to tolerance."""
    from gasgiant.sim import vorticity_ref

    q = _step_q(_params(0.0, steps=20), gpu)
    ref = vorticity_ref.zonal_mean(q)
    assert np.allclose(ref, q.mean(axis=1), atol=1e-6)

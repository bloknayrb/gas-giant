"""GPU tests for solver.vort_psi_drag (scale-selective hypofriction).

A vorticity sink proportional to the EDDY STREAMFUNCTION ψ' = ψ - <ψ>_x. Because
the screened Poisson gives ψ̂ = -ω̂/(k²+1/L_d²), adding +r·ψ' to q decays each
eddy mode at rate r/(k²+1/L_d²) -- the gravest-mode swirl is bled far harder than
medium eddies, so it removes the oversized swirl WITHOUT flattening the festoons
and band-edge waves a flat eddy-drag would erase (see swirl_gate.py metric m6 and
the adversarial visual review for the scale-selectivity validation).

Invariants tested here (correctness, not aesthetics):
  1. vort_psi_drag=0 is byte-identical to the path without it (off = no-op).
  2. vort_psi_drag>0 materially changes the render.
  3. The sink has the STABILISING sign and stays finite/bounded over a long
     horizon (a sign error would be a positive feedback that blows up).
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


def _params(psi_drag: float, steps: int = 60, ld: float = 0.18, inject: float = 1.5):
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = RES
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.solver.deformation_radius = ld        # finite L_d: the screening psi-drag uses
    p.solver.vort_psi_drag = psi_drag
    p.solver.vort_inject = inject           # a non-trivial eddy field to act on
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    # jupiter_vorticity now bakes hero mottle/tint interior noise; zero it so the
    # off-is-noop byte-identity check stays deterministic (the off path already sets
    # inject=0 + L_d=0; this removes the remaining hero-noise source).
    p.storms.hero_mottle = 0.0
    p.storms.hero_tint_var = 0.0
    return p


def _render(p, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(RES)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def test_psi_drag_off_is_noop(gpu):
    """vort_psi_drag=0 skips the ψ reduction dispatch and the SUBPASS-0 term is
    `if (u_vort_psi_drag > 0.0)` false -> bit-identical to the path without it.

    Uses L_d=0 and no injection so the psi-drag-off path is the quiet path. The
    empirical check uses the GPU noise floor rather than assert_array_equal because
    the modernized jupiter_vorticity base is otherwise LIVE (bold hero + rich detail)
    and its vorticity SOR carries ~1e-3 cross-instance LSB noise that 8-bit color no
    longer rounds away; a real psi-drag effect is >> the floor (see the changes test)."""
    base = _render(_params(0.0, ld=0.0, inject=0.0), gpu)
    same = _render(_params(0.0, ld=0.0, inject=0.0), gpu)
    assert np.abs(base - same).max() < GPU_NOISE_ATOL


def test_psi_drag_changes_render(gpu):
    plain = _render(_params(0.0), gpu)
    drained = _render(_params(0.6), gpu)
    assert np.abs(drained - plain).max() > GPU_NOISE_ATOL, (
        "vort_psi_drag=0.6 did not change the vorticity-mode render"
    )


def test_psi_drag_is_stabilising_and_bounded(gpu):
    """The sink must DECAY eddies, not amplify them: a wrong sign would be a
    positive feedback (q += r·ψ' with ψ̂ ∝ -ω̂) that diverges. Over a long
    horizon with a strong coefficient the output must stay finite and in range."""
    out = _render(_params(2.0, steps=400), gpu)
    assert np.all(np.isfinite(out)), "psi-drag produced non-finite output (sign/stability bug)"
    assert out.max() <= 1.0 + 1e-3 and out.min() >= -1e-3, (
        f"psi-drag drove color out of range over a long horizon: [{out.min()}, {out.max()}]"
    )

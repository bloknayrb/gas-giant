"""Localized eddy injection (solver.vort_inject_mask).

global = churn everywhere (legacy); belts/shear = localized so the
anticyclonic zones stay smooth. Two invariants matter:

  1. The mask must NEVER affect the inject=0 path (the shipped preset has
     vort_inject=0, so global/belts/shear must render identically there).
  2. With injection on, a localized mask must churn materially FEWER pixels
     than global -- that is the whole point of "don't churn the entire map".
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import InjectMask, SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2  # matches test_m3_ship: > session LSB noise, << inject signal
CHANGE_THRESH = 0.05   # per-pixel delta counting as "materially churned"


def _params(inject: float, mask: InjectMask):
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    p.solver.type = SolverType.VORTICITY
    p.solver.vort_inject = inject
    p.solver.vort_inject_mask = mask
    p.solver.vort_inject_scale = 2.0
    return p


def _render(p, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(512)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def _changed_fraction(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.abs(a - b).max(axis=-1)
    return float((delta > CHANGE_THRESH).mean())


def test_mask_is_noop_when_inject_zero(gpu):
    """vort_inject=0 -> the injection block is skipped, so the mask choice
    cannot change the render. Guards the shipped (inject=0) preset path."""
    base = _render(_params(0.0, InjectMask.GLOBAL), gpu)
    for mask in (InjectMask.BELTS, InjectMask.SHEAR):
        other = _render(_params(0.0, mask), gpu)
        maxdiff = np.abs(base - other).max()
        assert maxdiff <= GPU_NOISE_ATOL, f"mask={mask} changed inject=0 render (maxdiff={maxdiff})"


def test_localized_mask_churns_fewer_pixels_than_global(gpu):
    """With injection on, shear localization must leave materially more of the
    map untouched than global injection does."""
    baseline = _render(_params(0.0, InjectMask.GLOBAL), gpu)
    glob = _render(_params(0.8, InjectMask.GLOBAL), gpu)
    shear = _render(_params(0.8, InjectMask.SHEAR), gpu)

    global_churn = _changed_fraction(baseline, glob)
    shear_churn = _changed_fraction(baseline, shear)

    assert global_churn > 0.1, f"global injection should churn the map (frac={global_churn})"
    assert shear_churn < global_churn, (
        f"shear must localize: churned {shear_churn:.3f} vs global {global_churn:.3f}"
    )

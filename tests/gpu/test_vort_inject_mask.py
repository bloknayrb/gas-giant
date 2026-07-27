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


def test_localized_masks_churn_fewer_pixels_than_global(gpu):
    """With injection on, BOTH localized masks must leave materially more of the
    map untouched than global injection does -- and BELTS must be materially
    WIDER than SHEAR.

    The belts half was added with `green_giant`, the first factory preset to use
    that mask. Until then BELTS was exercised ONLY by the inject=0 no-op test
    above, i.e. only where it is guaranteed inert, while the localization
    assertion was written for SHEAR alone. A shipped preset now rests on the
    branch.

    The ORDERING (belts > shear) is the part worth pinning, because it is the
    sizing rule for `vort_inject`. A mask is a MULTIPLIER, and the two cover very
    different fractions of the map, so a shear-calibrated amplitude carried across
    to BELTS applies several times the integrated forcing -- the documented way to
    dissolve the banding. If a future change quietly narrowed belt_mask, the
    amplitude guidance in docs/formations.md and build_green_giant_preset.py would
    become wrong; this fails loudly instead.

    Measured on THIS test's layout (`jupiter_vorticity`, a 12-band template):
    belt_mask mean 0.556 with 55.6% of latitudes above 0.5, against shear_norm's
    0.096 and 4.8%. (green_giant's own 16-band seeded layout gives 0.475/47.6%
    and 0.110/3.9% -- same ordering, different numbers; quote whichever layout you
    actually measured.)

    All four renders live in one test rather than two: the GPU tier already runs
    >3 h under llvmpipe, and a separate belts test would re-render the shared
    baseline/global/shear trio for one new datapoint.
    """
    baseline = _render(_params(0.0, InjectMask.GLOBAL), gpu)
    glob = _render(_params(0.8, InjectMask.GLOBAL), gpu)
    shear = _render(_params(0.8, InjectMask.SHEAR), gpu)
    belts = _render(_params(0.8, InjectMask.BELTS), gpu)

    global_churn = _changed_fraction(baseline, glob)
    shear_churn = _changed_fraction(baseline, shear)
    belts_churn = _changed_fraction(baseline, belts)

    assert global_churn > 0.1, f"global injection should churn the map (frac={global_churn})"
    assert shear_churn < global_churn, (
        f"shear must localize: churned {shear_churn:.3f} vs global {global_churn:.3f}"
    )
    assert belts_churn > 0.0, (
        "BELTS injection changed nothing -- the mask is inert, not localized"
    )
    assert belts_churn < global_churn, (
        f"belts must localize: churned {belts_churn:.3f} vs global {global_churn:.3f}"
    )
    assert belts_churn > shear_churn, (
        f"belts must be WIDER than shear (the vort_inject sizing rule depends on "
        f"it): belts {belts_churn:.3f} vs shear {shear_churn:.3f}"
    )

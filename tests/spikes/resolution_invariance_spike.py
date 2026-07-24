"""Phase-0 calibration spike for resolution-invariant development.

Question: does the resolution-invariant scaling actually collapse the
cross-resolution drift -- i.e. does a run developed at a HIGHER resolution
(downsampled back) reproduce the reference-resolution run better with the flag
ON than OFF? And does it, as predicted, collapse for a NUDGE-dominated config
while only partially helping the TURBULENCE-dominated flagship?

Design (outside pytest testpaths; run explicitly, needs a GL 4.3 context):
  reference = 512. "truth" = develop at 512. "test" = develop at 1024, area-mean
  downsampled to the 512 grid. We compare truth vs test with the flag OFF (dev
  develops less far at 1024, rates unscaled) and ON (effective steps + scaled
  rates).

Two-sided metric (a one-sided metric has burned this project before):
  * zonal-mean tracer profile RMS across latitude  -> LARGE-SCALE / banding
  * whole-field Pearson correlation (downsampled)  -> eddy / storm PLACEMENT
Success = flag ON is closer (lower profile-RMS, higher correlation) than OFF,
strongly for the nudge-dominated config, at least partially for the flagship.

    uv run python tests/spikes/resolution_invariance_spike.py
"""

from __future__ import annotations

import numpy as np

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset

REF = 512
HIGH = 1024
DEV_STEPS = 120  # at the reference; the 1024 run does 2x with the flag on


def _nudge_dominated():
    """gas_giant_warm retuned to be nudge-dominated: kill the stochastic eddy
    injection and shorten the relaxation leash so the large scale is set by the
    painted template + nudge (the regime the scaling should fully collapse)."""
    p = load_factory_preset("gas_giant_warm")
    p.solver.vort_inject = 0.0
    p.solver.vort_relax_tau = 40.0
    p.storms.wake_turbulence = 0.0
    return p


def _turbulence_dominated():
    """Stock gas_giant_warm (vort_inject 1.8, weak relax) -- the flagship the
    honest-scope note says stays only partially invariant."""
    return load_factory_preset("gas_giant_warm")


def _develop(gpu, make_params, *, resolution, invariant):
    p = make_params()
    p.sim.resolution = resolution
    p.sim.dev_steps = DEV_STEPS
    p.sim.resolution_invariant = invariant
    p.sim.reference_resolution = REF
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=60)
    arr = np.array(sim.tracers.read_current(), dtype=np.float64)
    sim._release_sim()
    return arr


def _channel0(arr):
    return arr[..., 0] if arr.ndim == 3 else arr


def _downsample2x(field):
    """Area-mean 2x reduction (H, W) -> (H/2, W/2)."""
    h, w = field.shape
    return field[: h // 2 * 2, : w // 2 * 2].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def _zonal_profile(field):
    """Mean over longitude at each latitude -> the banding signature."""
    return field.mean(axis=field.ndim - 1) if field.shape[-1] > field.shape[0] \
        else field.mean(axis=0)


def _metrics(truth, test_down):
    # Align shapes defensively (rounding can differ by a row/col).
    h = min(truth.shape[0], test_down.shape[0])
    w = min(truth.shape[1], test_down.shape[1])
    a, b = truth[:h, :w], test_down[:h, :w]
    prof_rms = float(np.sqrt(np.mean((a.mean(axis=1) - b.mean(axis=1)) ** 2)))
    af, bf = a.ravel() - a.mean(), b.ravel() - b.mean()
    denom = np.sqrt((af @ af) * (bf @ bf))
    corr = float((af @ bf) / denom) if denom > 0 else float("nan")
    return prof_rms, corr


def _run_case(gpu, name, make_params):
    truth = _channel0(_develop(gpu, make_params, resolution=REF, invariant=False))
    results = {}
    for flag in (False, True):
        high = _channel0(_develop(gpu, make_params, resolution=HIGH, invariant=flag))
        down = _downsample2x(_orient(high))
        results[flag] = _metrics(_orient(truth), down)
    print(f"\n=== {name} ===")
    print(f"  {'flag':<8}{'profile_RMS':>14}{'correlation':>14}")
    for flag in (False, True):
        rms, corr = results[flag]
        print(f"  {'ON' if flag else 'OFF':<8}{rms:>14.5f}{corr:>14.4f}")
    rms_off, corr_off = results[False]
    rms_on, corr_on = results[True]
    print(f"  -> profile_RMS {'IMPROVED' if rms_on < rms_off else 'WORSE'} "
          f"({rms_off:.5f} -> {rms_on:.5f}); "
          f"correlation {'IMPROVED' if corr_on > corr_off else 'WORSE'} "
          f"({corr_off:.4f} -> {corr_on:.4f})")
    return results


def _orient(field):
    """Ensure (H, W) with H < W (equirect 2:1). read_current may return (W,H)."""
    return field.T if field.shape[0] > field.shape[1] else field


def main():
    gpu = GpuContext.headless()
    gpu.make_current()
    _run_case(gpu, "nudge-dominated (vort_inject=0, tau=40)", _nudge_dominated)
    _run_case(gpu, "turbulence-dominated (stock gas_giant_warm)", _turbulence_dominated)


if __name__ == "__main__":
    main()

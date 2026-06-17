"""M0.5 512x256 spin-up driver with physical-time forcing rescale.

Usage
-----
    uv run python scripts/swp_spinup.py [--steps N] [--nu4-probe] [--nu4 FLOAT]

Workflow
--------
1.  Compute dt_ref from the 192x96 emergent_init (the M0 reference resolution).
2.  Init 512x256 with emergent_init and compute dt_hi; dt_scale = dt_hi / dt_ref.
3.  Short nu4 stability probe: run 1000 steps each for nu4 in {0.05, 0.07, 0.10};
    pick the highest nu4 that stays finite and shows the most eddy activity.
4.  Full spin-up to target ~32k steps (or --steps N), printing eddy_vorticity_std
    and ms/step every 1000 steps.
5.  Report: eddy_vort_std progression, whether >=1.0 reached and at which step,
    chosen nu4, ms/step, projected wall time for 32k.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _compute_dt(W: int, H: int, geff: float = 1.0, h_mean: float = 5.0) -> float:
    from gasgiant.sim.sw_spike.grid import Grid
    g = Grid(W, H)
    c_gw = np.sqrt(geff * h_mean)
    dx_min = min(g.cos_c.min() * g.dlam, g.dphi)
    return 0.3 * dx_min / c_gw


def nu4_probe(gpu, W_hi: int, H_hi: int, dt_scale: float, steps: int = 1000) -> float:
    """Try nu4 in {0.05, 0.07, 0.10}.  Return best (highest finite + richest) value."""
    import dataclasses

    from gasgiant.sim.sw_gpu_probe import solver as gsolver
    from gasgiant.sim.sw_spike import init

    candidates = [0.05, 0.07, 0.10]
    best_nu4 = candidates[0]
    best_evs = -1.0

    print("\n-- nu4 stability probe ---")
    for nu4 in candidates:
        st_test = init.emergent_init(
            W=W_hi, H=H_hi, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4,
        )
        st_test = dataclasses.replace(st_test, nu4=nu4)
        sg = gsolver.SwpSolver.from_cpu_state(gpu, st_test, forcing_dt_scale=dt_scale)
        for _ in range(steps):
            sg.step()
        h1 = sg.download("h1")
        finite = bool(np.all(np.isfinite(h1)))
        evs = sg.eddy_vorticity_std() if finite else float("nan")
        status = "STABLE" if finite else "BLEW UP"
        print(f"  nu4={nu4:.2f}: {status}  eddy_vort_std={evs:.4f}")
        if finite and evs > best_evs:
            best_evs = evs
            best_nu4 = nu4
        del sg  # release GPU textures

    print(f"  -> chose nu4={best_nu4:.2f} (richest stable)\n")
    return best_nu4


def main() -> None:
    parser = argparse.ArgumentParser(description="M0.5 512x256 spin-up")
    parser.add_argument("--steps", type=int, default=8000,
                        help="total GPU steps (default 8000; use 32000 for full run)")
    parser.add_argument("--nu4",   type=float, default=None,
                        help="skip nu4 probe and use this value directly")
    parser.add_argument("--nu4-probe",    dest="nu4_probe", action="store_true",  default=True)
    parser.add_argument("--no-nu4-probe", dest="nu4_probe", action="store_false")
    args = parser.parse_args()

    print("=== M0.5 512x256 spin-up ============================================")

    import dataclasses

    from gasgiant.gl.context import GpuContext
    from gasgiant.sim.sw_gpu_probe import solver as gsolver
    from gasgiant.sim.sw_spike import init

    W_ref, H_ref = 192, 96
    W_hi,  H_hi  = 512, 256

    # dt_ref: M0 reference timescale from 192x96 emergent_init
    st_ref = init.emergent_init(
        W=W_ref, H=H_ref, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4,
    )
    dt_ref = st_ref.dt
    print(f"dt_ref (192x96)  = {dt_ref:.6e}")

    dt_hi    = _compute_dt(W_hi, H_hi)
    dt_scale = dt_hi / dt_ref
    print(f"dt_hi  (512x256) = {dt_hi:.6e}")
    print(f"dt_scale         = {dt_scale:.6f}  (dt_hi / dt_ref)")
    equiv_ref_steps = args.steps * dt_scale
    print(f"Physical equiv.  = {equiv_ref_steps:.0f} ref-steps for {args.steps} hi-res steps")

    gpu = GpuContext.headless()

    # nu4 selection
    if args.nu4 is not None:
        chosen_nu4 = args.nu4
        print(f"\nnu4 override: {chosen_nu4:.2f}")
    elif args.nu4_probe:
        chosen_nu4 = nu4_probe(gpu, W_hi, H_hi, dt_scale, steps=1000)
    else:
        chosen_nu4 = 0.07
        print(f"\nnu4 probe skipped; using default {chosen_nu4}")

    # Full spin-up
    print("-- Full spin-up ---")
    st_hi = init.emergent_init(
        W=W_hi, H=H_hi, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4,
    )
    st_hi = dataclasses.replace(st_hi, nu4=chosen_nu4)
    sg = gsolver.SwpSolver.from_cpu_state(gpu, st_hi, forcing_dt_scale=dt_scale)

    TOTAL  = args.steps
    CHUNK  = 1000
    TARGET = 1.0

    reached_target_at: int | None = None
    evs_history: list[tuple[int, float]] = []

    evs0 = sg.eddy_vorticity_std()
    evs_history.append((0, evs0))
    print(f"  step={0:6d}  eddy_vort_std={evs0:.4f}")

    t_run_start = time.perf_counter()
    step_count  = 0
    ms_per_step_last = float("nan")

    for chunk_start in range(0, TOTAL, CHUNK):
        n = min(CHUNK, TOTAL - chunk_start)
        t0 = time.perf_counter()
        for _ in range(n):
            sg.step()
        t1 = time.perf_counter()
        step_count += n

        h1 = sg.download("h1")
        if not np.all(np.isfinite(h1)):
            print(f"  !! BLOWUP at step {step_count} -- aborting.")
            break

        evs = sg.eddy_vorticity_std()
        ms_per_step_last = (t1 - t0) / n * 1000.0
        print(f"  step={step_count:6d}  eddy_vort_std={evs:.4f}  {ms_per_step_last:.2f} ms/step")
        evs_history.append((step_count, evs))

        if reached_target_at is None and evs >= TARGET:
            reached_target_at = step_count
            print(f"  *** eddy_vorticity_std >= {TARGET:.1f} reached at step {step_count} ***")

    t_total = time.perf_counter() - t_run_start
    last_evs = evs_history[-1][1] if evs_history else float("nan")
    ms_avg   = t_total / step_count * 1000.0 if step_count > 0 else float("nan")
    proj_32k = ms_avg * 32000 / 1000.0

    print("\n=== Summary ==========================================================")
    print(f"Resolution       : {W_hi}x{H_hi}")
    print(f"dt_scale         : {dt_scale:.6f}  (dt_hi/dt_ref)")
    print(f"Chosen nu4       : {chosen_nu4:.2f}")
    print(f"Total steps run  : {step_count}")
    print(f"Final evs        : {last_evs:.4f}")
    if reached_target_at is not None:
        print(f"evs >= {TARGET:.1f} at step : {reached_target_at}")
    else:
        print(f"evs >= {TARGET:.1f}         : NOT REACHED in {step_count} steps")
    print(f"Wall time        : {t_total:.1f} s  ({step_count} steps)")
    print(f"ms/step (avg)    : {ms_avg:.2f}")
    print(f"Projected 32k    : {proj_32k:.0f} s  = {proj_32k / 60:.1f} min")


if __name__ == "__main__":
    main()

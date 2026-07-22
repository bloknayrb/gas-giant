"""Phase 0b (perf plan): attribute the per-step cost of a vorticity dev run and
size the SOR Poisson solve's share -- the data that gates the optional SOR
convergence tuning (Phase 3).

Method (no solver edits, no fragile internals): warm one gas_giant_warm sim into
its steady vorticity solve, then time equal batches of steps at several
``solver.poisson_iters`` values. ``poisson_iters`` is read live inside
``Solver.step`` (solver.py), so it can be toggled on the SAME warmed sim between
batches -- which controls for warm-start state and isolates the SOR cost as a
clean difference:

    per-iter SOR cost  = (t_steps[hi] - t_steps[lo]) / (hi - lo)
    SOR cost @ N iters = N * per-iter cost
    SOR share @ N      = SOR cost @ N / t_steps[N]

Everything else in the step (omega advection, recover, analytic psi, warm-start
copy, feather, velocity, MacCormack, exchange) is the poisson_iters-independent
remainder. SOR runs on all three domains (equirect + 2 poles), so the measured
per-iter cost already sums the domains.

The SOR *share* is ~resolution-independent ONLY because the remainder is itself
GPU-compute-bound (near-zero CPU/submission overhead) -- confirmed by
perf_step_breakdown.py, which measures the step ~100% GPU-bound and the share
near-constant (28% @2048 vs 26% @4096). So a tractable --res gives a valid share
even though absolute per-step ms is lower than the 4096 flagship. (If a future
change made the step submission-bound, a lower --res would understate the share.)
Run:

    uv run python scripts/perf_sor_timing.py                 # preset resolution
    uv run python scripts/perf_sor_timing.py --res 2048      # faster, same share
"""

from __future__ import annotations

import argparse
import statistics
import time

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset

ITER_POINTS = [8, 48, 96]  # 48 = shipped default; 8 = floor; 96 brackets the linear fit


def _fence(sim: Simulation) -> None:
    """Force GPU completion by reading the current tracer back (a sync point)."""
    sim.tracers.read_current()


def _time_steps(sim: Simulation, n_iters: int, steps: int, repeats: int) -> float:
    """Median wall-clock seconds for one `steps`-batch at this poisson_iters."""
    sim.solver.params.solver.poisson_iters = n_iters
    sim.solver.step(2)  # settle at the new iter count
    _fence(sim)
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sim.solver.step(steps)
        _fence(sim)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="gas_giant_warm")
    ap.add_argument("--res", type=int, default=0, help="override sim.resolution (0 = preset)")
    ap.add_argument("--warmup", type=int, default=200, help="steps to reach the steady solve")
    ap.add_argument("--steps", type=int, default=60, help="steps per timed batch")
    ap.add_argument("--repeats", type=int, default=3, help="batches per iter point (median)")
    args = ap.parse_args()

    p = load_factory_preset(args.preset)
    if p.solver.type != SolverType.VORTICITY:
        raise SystemExit(f"{args.preset} is not a vorticity preset; SOR timing needs one")
    if args.res:
        p.sim.resolution = args.res
    # Uncap the dev run so warmup + timed batches aren't clamped by dev_steps.
    p.sim.dev_steps = args.warmup + args.steps * args.repeats * len(ITER_POINTS) + 100
    res = p.sim.resolution
    shipped_iters = p.solver.poisson_iters
    shipped_omega = p.solver.sor_omega

    gpu = GpuContext.headless()
    gpu.make_current()
    sim = Simulation(p, gpu)
    sim.solver.step(args.warmup)  # reach the steady vorticity solve
    _fence(sim)

    print(f"preset={args.preset} res={res} shipped: poisson_iters={shipped_iters} "
          f"sor_omega={shipped_omega}")
    print(f"warmup={args.warmup} steps  timed batch={args.steps} steps x {args.repeats} repeats\n")
    print(f"{'poisson_iters':>14} {'ms/step':>10}")
    per_step: dict[int, float] = {}
    for n_iters in ITER_POINTS:
        dt = _time_steps(sim, n_iters, args.steps, args.repeats)
        per_step[n_iters] = 1000.0 * dt / args.steps
        print(f"{n_iters:>14} {per_step[n_iters]:>10.2f}")

    # Linear fit across the bracketing iter points; the share denominator is the
    # SHIPPED step cost (not a hardcoded 48), so this holds if a preset ships a
    # different poisson_iters.
    if shipped_iters not in per_step:
        raise SystemExit(f"shipped poisson_iters={shipped_iters} not among measured ITER_POINTS={ITER_POINTS}")
    lo, hi = ITER_POINTS[0], ITER_POINTS[-1]
    per_iter_ms = (per_step[hi] - per_step[lo]) / (hi - lo)
    step_ms = per_step[shipped_iters]

    print()
    # A negative fit means SOR is below run-to-run noise at this res/steps (the
    # vorticity SOR carries documented LSB/thread divergence) -- report it as
    # unresolved rather than printing a negative share as if it were a fact.
    if per_iter_ms <= 0.0:
        print(f"SOR below measurement noise at res={res}/steps={args.steps} "
              f"(per-iter fit {per_iter_ms:.4f} ms <= 0); increase --res or --steps. Share not reported.")
    else:
        sor_ms = per_iter_ms * shipped_iters
        share = sor_ms / step_ms if step_ms else 0.0
        print(f"per-SOR-iter cost (all 3 domains) = {per_iter_ms:.3f} ms/iter")
        print(f"at shipped {shipped_iters} iters: SOR = {sor_ms:.2f} ms/step "
              f"({share * 100:.0f}% of step); remainder = {step_ms - sor_ms:.2f} ms/step")
        print()
        print("Phase 3 reads this as: SOR share sets the ceiling on any (omega, iters) "
              "cut; the remainder is what stays even at poisson_iters -> 0.")

    sim._release_sim()


if __name__ == "__main__":
    main()

"""Phase 0b follow-on (perf plan): split the vorticity per-step cost into GPU
compute vs CPU/submission overhead, and SOR vs the rest -- to learn what the
large "remainder" (left after SOR, see perf_sor_timing.py) actually is.

Two orthogonal cuts on one warmed gas_giant_warm sim (no solver edits):

1. GPU vs CPU. Wrap solver.step(batch) in a GL_TIME_ELAPSED query (pure GPU
   time) and also wall-clock it with a fence. wall - gpu is the CPU/submission
   overhead that isn't overlapped by the GPU (per-dispatch Python uniform sets,
   the ~333 driver submissions, advance_registry + pack_ssbo). If gpu ~= wall
   the step is compute-bound (remainder is real advection compute -> algorithmic,
   hard); if gpu << wall it is submission-bound (remainder is overhead ->
   batchable).

2. SOR vs rest, in *GPU* time. poisson_iters is read live inside step(), so the
   gpu-time difference between 8 and 48 iters isolates SOR's GPU cost; the rest
   of the GPU time is the advection/velocity/feather/exchange compute.

Also times the CPU prologue (advance_registry + pack_ssbo + upload) directly, to
see how much of the CPU overhead is those big per-step chunks vs diffuse
per-dispatch Python. Run:

    uv run python scripts/perf_step_breakdown.py --res 2048
    uv run python scripts/perf_step_breakdown.py --res 4096
"""

from __future__ import annotations

import argparse
import statistics
import time

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.advance import advance_registry


def _measure(sim: Simulation, n_iters: int, steps: int, repeats: int) -> tuple[float, float]:
    """(wall ms/step, gpu ms/step) at this poisson_iters, median of repeats."""
    sim.solver.params.solver.poisson_iters = n_iters
    sim.solver.step(2)  # settle at the new iter count
    sim.tracers.read_current()
    walls: list[float] = []
    gpus: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        q = sim.gpu.ctx.query(time=True)
        with q:
            sim.solver.step(steps)
        gpu_ms = q.elapsed / 1e6  # reading .elapsed fences on the scoped GPU work
        walls.append((time.perf_counter() - t0) * 1000.0)
        gpus.append(gpu_ms)
    return statistics.median(walls) / steps, statistics.median(gpus) / steps


def _time_cpu_prologue(sim: Simulation, reps: int) -> float:
    """ms/step for the CPU prologue the solver runs each step before dispatch.

    Mirrors Solver.step's prologue faithfully -- advance_registry with an
    ADVANCING step index (event scheduling keys off it) plus the SSBO
    orphan-on-grow guard -- so the measured cost is representative and can't
    raise on a registry that outgrows the buffer.
    """
    s = sim.solver
    idx = s.step_index
    t0 = time.perf_counter()
    for _ in range(reps):
        advance_registry(s.vortices, s.profiles, s.dt, idx,
                         s.events, s.params.storms)
        data = s.vortices.pack_ssbo()
        if data.nbytes > s._ssbo.size:
            s._ssbo.orphan(data.nbytes)
        s._ssbo.write(data.tobytes())
        idx += 1
    return (time.perf_counter() - t0) * 1000.0 / reps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="gas_giant_warm")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    p = load_factory_preset(args.preset)
    if p.solver.type != SolverType.VORTICITY:
        raise SystemExit(f"{args.preset} is not a vorticity preset")
    p.sim.resolution = args.res
    p.sim.dev_steps = args.warmup + args.steps * args.repeats * 3 + 100
    shipped = p.solver.poisson_iters
    if shipped <= 8:
        raise SystemExit(f"{args.preset} ships poisson_iters={shipped}; need >8 for the 8-vs-shipped fit")

    gpu = GpuContext.headless()
    gpu.make_current()
    sim = Simulation(p, gpu)
    sim.solver.step(args.warmup)
    sim.tracers.read_current()

    wall8, gpu8 = _measure(sim, 8, args.steps, args.repeats)
    wall48, gpu48 = _measure(sim, shipped, args.steps, args.repeats)
    prologue = _time_cpu_prologue(sim, args.steps * args.repeats)

    # GL_TIME_ELAPSED returns 0 on stacks without real timer queries (software
    # rasterizers -- llvmpipe/OSMesa). Bail loudly rather than emit a confident
    # but fabricated "submission-bound" verdict (gpu~0 -> cpu_overhead~wall).
    if gpu48 <= 0.0 or gpu8 <= 0.0:
        raise SystemExit("GL_TIME_ELAPSED returned 0 -- timer queries unsupported on this GL "
                         "stack (software renderer?); the GPU/CPU split is unavailable here")

    cpu_overhead = wall48 - gpu48

    print(f"preset={args.preset} res={args.res} poisson_iters={shipped}")
    print(f"warmup={args.warmup}  batch={args.steps} x {args.repeats}\n")
    print(f"  wall/step         {wall48:8.2f} ms")
    print(f"  gpu/step          {gpu48:8.2f} ms   ({gpu48 / wall48 * 100:4.0f}% of wall)")
    print(f"  cpu+submission    {cpu_overhead:8.2f} ms   ({cpu_overhead / wall48 * 100:4.0f}% of wall)")
    print(f"    of which prologue {prologue:6.2f} ms   (advance_registry + pack_ssbo + upload)")
    print()
    verdict = "COMPUTE-bound (remainder is real GPU work)" if gpu48 > cpu_overhead \
        else "SUBMISSION-bound (remainder is CPU/dispatch overhead -> batchable)"
    print(f"  verdict: {verdict}")

    # SOR vs rest, in GPU time. A negative fit means SOR is below run-to-run
    # noise at this res/steps (the vorticity SOR carries documented LSB/thread
    # divergence) -- report it as unresolved instead of printing a negative share.
    per_iter_gpu = (gpu48 - gpu8) / (shipped - 8)
    print()
    if per_iter_gpu <= 0.0:
        print(f"  GPU split: SOR below measurement noise at res={args.res}/steps={args.steps} "
              f"(per-iter fit {per_iter_gpu:.4f} ms <= 0); increase --res or --steps.")
    else:
        sor_gpu = per_iter_gpu * shipped
        nonsor_gpu = gpu48 - sor_gpu
        print(f"  GPU split @ {shipped} iters:")
        print(f"    SOR             {sor_gpu:8.2f} ms   ({sor_gpu / gpu48 * 100:4.0f}% of gpu)")
        print(f"    non-SOR (advect/vel/feather/exchange) {nonsor_gpu:6.2f} ms   "
              f"({nonsor_gpu / gpu48 * 100:4.0f}% of gpu)")

    sim._release_sim()


if __name__ == "__main__":
    main()

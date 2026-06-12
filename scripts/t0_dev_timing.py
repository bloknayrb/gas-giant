"""T0 (v1.5): dev-phase timing at sim.resolution {2048,3072,4096} x dev_steps {500,800}.

Times the DEVELOPMENT PHASE alone (solver stepping) in isolation -- NOT the
tiled derive, which is sim-resolution-independent (reads only snapshot
textures). Projects 16K all-on totals arithmetically from the measured
dev cost, anchored on the recorded v1.4 baseline:
    total(16K all-on) = 39.0 s = dev(500 @ 2048) + tiled_derive
so tiled_derive = 39.0 - dev(500 @ 2048), and
    total(res, steps) = tiled_derive + dev(steps @ res).

ONE GpuContext per process (shared across configs). Run:
    uv run python scripts/t0_dev_timing.py
"""

from __future__ import annotations

import time

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset

RESOLUTIONS = [2048, 3072, 4096]
STEPS = [500, 800]
GATE = 40.0
V14_TOTAL_16K = 39.0  # recorded median-of-3, all-on, dev=500@2048 (realism.md)


def time_dev(gpu: GpuContext, resolution: int, dev_steps: int) -> float:
    """Wall-clock seconds for the development run at this resolution.

    Build the Simulation (constructs solver + init_tracers, no stepping), then
    time only solver.step(dev_steps) with a GPU finish via a tiny readback.
    """
    p = load_factory_preset("jupiter_like")
    p.sim.resolution = resolution
    p.sim.dev_steps = dev_steps
    sim = Simulation(p, gpu)
    # warm: tracers already initialised in _build; step once to compile/warm
    sim.solver.step(1)
    t0 = time.perf_counter()
    sim.solver.step(dev_steps)
    # force completion: read tracers back (GPU fence)
    _ = sim.tracers.read_current()
    dt = time.perf_counter() - t0
    sim._release_sim()
    return dt


def main() -> None:
    gpu = GpuContext.headless()
    gpu.make_current()
    results: dict[tuple[int, int], float] = {}
    print(f"{'res':>6} {'steps':>6} {'dev_s':>8} {'per_step_ms':>12}")
    for res in RESOLUTIONS:
        for steps in STEPS:
            dev = time_dev(gpu, res, steps)
            results[(res, steps)] = dev
            print(f"{res:>6} {steps:>6} {dev:>8.2f} {1000 * dev / steps:>12.2f}")

    # Anchor: tiled derive = v1.4 16K total - dev(500 @ 2048)
    base = results[(2048, 500)]
    tiled_derive = V14_TOTAL_16K - base
    print()
    print(f"baseline dev(500@2048) = {base:.2f}s  ->  implied tiled_derive = {tiled_derive:.2f}s")
    print()
    print(f"{'res':>6} {'steps':>6} {'proj_16K_total_s':>18} {'vs 40s gate':>14}")
    for res in RESOLUTIONS:
        for steps in STEPS:
            total = tiled_derive + results[(res, steps)]
            verdict = "OK" if total <= GATE else "OVER"
            print(f"{res:>6} {steps:>6} {total:>18.2f} {verdict:>14}")


if __name__ == "__main__":
    main()

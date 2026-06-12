"""v1.5 acceptance perf gate: 16K all-on tiled export, median-of-3 wall time.

v1.4 baseline was 39.0 s median vs the 40 s gate. v1.5 adds per-step belt
replenishment (P1) and a second render-side fold backtrace (P2); this verifies
the tuned jupiter_like still fits the gate.

    uv run python scripts/perf_16k.py
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

from gasgiant.engine.facade import Simulation
from gasgiant.export.exporter import run_export
from gasgiant.gl import GpuContext
from gasgiant.params.presets import load_factory_preset

WIDTH = 16384
RUNS = 3
GATE = 40.0


def main() -> None:
    gpu = GpuContext.headless()
    gpu.make_current()
    times = []
    for i in range(RUNS):
        p = load_factory_preset("jupiter_like")
        sim = Simulation(p, gpu)
        with tempfile.TemporaryDirectory() as td:
            t0 = time.perf_counter()
            run_export(sim, Path(td), WIDTH)
            dt = time.perf_counter() - t0
        times.append(dt)
        sim._release_sim()
        print(f"run {i + 1}: {dt:.2f}s")
    med = statistics.median(times)
    print(f"\n16K all-on median-of-{RUNS}: {med:.2f}s  (gate {GATE}s)  -> "
          f"{'OK' if med <= GATE else 'OVER'}")


if __name__ == "__main__":
    main()

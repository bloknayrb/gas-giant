"""Spin up the M0 shallow-water spike and report equilibration vs steps."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gasgiant.sim.sw_spike import init, solver  # noqa: E402


def spin_up(steps=3000, W=384, H=192, log_every=200):
    st = init.emergent_init(W=W, H=H, f0=4.0, gp=(1.0, 0.05),
                            n_bands=18, band_contrast=0.5)
    t0 = time.perf_counter()
    series = []
    for s in range(steps):
        st = solver.step(st, dt=st.dt)
        if s % log_every == 0:
            z = float(np.std(solver.relative_vorticity_top(st)))
            series.append((s, z))
            print(f"step {s:5d}  vort_std {z:.4f}")
            assert np.all(np.isfinite(st.h1)), f"NaN at step {s}"
    dt_wall = time.perf_counter() - t0
    print(f"spin-up {steps} steps in {dt_wall:.1f}s ({1000*dt_wall/steps:.1f} ms/step)")
    finalz = series[-1][1]
    eq_step = next((s for s, z in series if z >= 0.9 * finalz), steps)
    print(f"~equilibration step: {eq_step}")
    return st


if __name__ == "__main__":
    spin_up()

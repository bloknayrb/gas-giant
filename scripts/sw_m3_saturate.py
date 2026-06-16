"""M3 SATURATION FINDER — anchor on the VALIDATED crux config, find the
renderable finite-amplitude window.

The crux test PROVES growth at: W=192,H=96, gp1=0.05, gp2=0.3, xi=3, pert=1e-3,
dt_safety=0.3, nu4=0, NO forcing -> sigma=2.77e-5/s (R2=1.0), saturating by
lower-layer outcrop. This script replays that exact config (variant A) plus two
deltas (B: bigger seed, C: smaller dt) to isolate why the sweep variants decayed,
and tracks the metrics that matter for the RENDER:
  - eddy_interface_var growth (the mode)
  - top-layer eddy Rossby Ro1 (what the render's non-vacuity gate measures)
  - min(h2) (how close to outcrop / crash)
Stops at outcrop (ValueError) or step cap, reporting the step where Ro1 first
crosses 0.10 (renderable finite amplitude) and the pre-outcrop snapshot.

Usage: py -3 scripts/sw_m3_saturate.py <variant A|B|C>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402

OUT = Path("out/audit/m3")
W_GRID, H_GRID = 192, 96
RO_TARGET = 0.10
CHUNK = 250
CAP = 14000

VARIANTS = {
    # name: (pert_amp_frac, dt_safety)  -- all gp1=0.05, xi=3, nu4=0, no forcing
    "A": (1e-3, 0.30),   # EXACT crux test config (known-good)
    "B": (5e-3, 0.30),   # bigger seed (isolate pert_amp_frac)
    "C": (1e-3, 0.18),   # smaller dt (isolate dt_safety)
}


def eddy_rossby(u, v, g, f0):
    zeta = ref.vorticity(u, v, g)
    eddy = zeta - zeta.mean(axis=1, keepdims=True)
    return float(np.std(eddy)) / abs(f0)


def main(variant: str):
    pert, dts = VARIANTS[variant]
    st = ref.baroclinic_test_state(
        W=W_GRID, H=H_GRID, unstable=True, seed=0,
        gp1=0.05, gp2=0.3, xi_unstable=3.0, pert_amp_frac=pert,
        dt_safety=dts, nu4=0.0,
    )
    # NO forcing: leave tau_rad/tau_drag/sponge at Sw2State defaults (0 -> no-op).
    g = st.g
    f0 = 2.0 * st.omega * np.sin(np.radians(45.0))
    iv0 = ref.eddy_interface_var(st) + 1e-30
    print(f"[{variant}] pert={pert} dt_safety={dts} dt={st.dt:.3e} ivar0={iv0:.3e} "
          f"shear={st._shear:.2f}", flush=True)

    reached = None
    outcrop_step = None
    snap = None        # (step, Ro1, Ro2, ivar_x) just before outcrop
    steps = 0
    t0 = time.perf_counter()
    while steps < CAP:
        try:
            for _ in range(CHUNK):
                ref.step_2layer(st)
        except (ValueError, AssertionError) as ex:
            outcrop_step = steps + CHUNK
            print(f"[{variant}] OUTCROP/trap ~step {outcrop_step}: {str(ex)[:55]}", flush=True)
            break
        steps += CHUNK
        ro1 = eddy_rossby(st.u1, st.v1, g, f0)
        ro2 = eddy_rossby(st.u2, st.v2, g, f0)
        ivar_x = (ref.eddy_interface_var(st) + 1e-30) / iv0
        minh2 = float(st.h2.min())
        snap = (steps, ro1, ro2, ivar_x, minh2)
        ms = (time.perf_counter() - t0) / steps * 1000.0
        print(f"[{variant}] step={steps:6d} Ro1={ro1:.4f} Ro2={ro2:.4f} "
              f"ivar_x={ivar_x:.2e} min_h2={minh2:.1f} {ms:.1f}ms/s", flush=True)
        if reached is None and ro1 >= RO_TARGET:
            reached = steps
            print(f"[{variant}] *** TOP-layer Ro1>={RO_TARGET} (renderable) at step {steps} ***", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    line = (f"variant {variant}: pert={pert} dt_safety={dts} | "
            f"Ro1>=0.10@{reached} outcrop@{outcrop_step} "
            f"last={snap}")
    with open(OUT / "saturate_results.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print("\nRESULT: " + line, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

"""M3 baroclinic SATURATION SWEEP (CPU, low-res, fast).

Purpose: the render gate produces LAMINAR output because per-step damping
(tau_rad thermal relax 1/4000, tau_drag bottom drag 1/12000) is FASTER than the
baroclinic growth rate (1/efold ~ 1/52000 at 256x128). The seed decays instead
of growing. This sweep tests, cheaply on CPU at low res, which (xi, damping)
combinations let the validated instability actually SATURATE into finite eddies
(eddy Rossby Ro >= 0.10) BEFORE we spend GPU time on a long render run.

One config per invocation (argv[1] = config index). Each appends a one-line
verdict to out/audit/m3/sweep_results.txt so configs can run in parallel.

Metric: eddy Rossby = std(non-zonal relative vorticity) / |f0(45deg)|, measured
on BOTH layers. The gate renders the TOP layer, but the instability lives in the
LOWER layer; we report both so we can see the mode grow and couple upward.

Usage: py -3 scripts/sw_m3_sweep.py <config_index>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402

OUT = Path("out/audit/m3")
RESULTS = OUT / "sweep_results.txt"

# Resolution: the crux gate (test_m3_baroclinic) proved growth at 192x96, tuned
# so L_D ~ 3 cells. 96x48 UNDER-resolves L_D (~1.5 cells) and the mode cannot
# form (confirmed: 8 e-foldings produced zero growth). 192x96 is the validated
# grid; below it the instability is a numerical artifact of under-resolution.
W_GRID, H_GRID = 192, 96
RO_TARGET = 0.10
STEP_CHUNK = 500
MAX_STEPS = 80000          # hard cap; high-xi configs early-stop well before this

# Each config: (name, gp1, xi, tau_rad, tau_drag, sponge, nu4, pert).
# tau_* = 0.0 turns that damping OFF (guarded in apply_forcing).
#
# KEY: the CRUX gate (test_m3_baroclinic) that PROVED clean exponential growth
# to saturation used gp1=0.05 and NO forcing. The render gate diverged to
# gp1=0.5 + thermal/drag forcing -> the validated instability never ran. These
# configs walk back from the render config toward the validated crux config.
CONFIGS = [
    # name              gp1    xi   tau_rad  tau_drag  sponge  nu4   pert
    ("render_cfg",      0.50,  3.0,  4000.0,  12000.0,  0.04,  0.06, 5e-3),  # the laminar render config (baseline)
    ("crux_repro",      0.05,  3.0,     0.0,      0.0,  0.00,  0.00, 5e-3),  # validated crux: gp1=.05, no forcing
    ("crux_sponge",     0.05,  3.0,     0.0,      0.0,  0.02,  0.06, 5e-3),  # + min sponge/nu4 for render stability
    ("crux_sustained",  0.05,  3.0, 200000.0, 200000.0,  0.02,  0.06, 5e-3),  # damping >> efold (statistically steady)
    ("crux_xi4",        0.05,  4.0,     0.0,      0.0,  0.02,  0.06, 5e-3),  # slightly hotter (xi6 blew up @96x48)
    ("lowgp_render",    0.05,  3.0,  4000.0,  12000.0,  0.04,  0.06, 5e-3),  # render forcing but gp1=.05 (isolate gp1)
]


def make_state(cfg):
    _, gp1, xi, tau_rad, tau_drag, sponge, nu4, pert = cfg
    st = ref.baroclinic_test_state(
        W=W_GRID, H=H_GRID, unstable=True, seed=0, nu4=nu4, gp1=gp1,
        xi_unstable=xi, pert_amp_frac=pert, dt_safety=0.18,
    )
    st.tau_rad = tau_rad
    st.tau_drag = tau_drag
    st.sponge_rate = sponge
    st.h_eq1 = st.h1.copy()
    st.h_eq2 = st.h2.copy()
    return st


def eddy_rossby(u, v, g, f0):
    zeta = ref.vorticity(u, v, g)
    eddy = zeta - zeta.mean(axis=1, keepdims=True)
    return float(np.std(eddy)) / abs(f0)


def run_config(idx: int):
    cfg = CONFIGS[idx]
    name = cfg[0]
    st = make_state(cfg)
    g = st.g
    f0 = 2.0 * st.omega * np.sin(np.radians(45.0))
    efold = ref.efold_steps_estimate(st)
    budget = int(min(MAX_STEPS, max(6 * efold, 20000)))

    ro1_0 = eddy_rossby(st.u1, st.v1, g, f0)
    ro2_0 = eddy_rossby(st.u2, st.v2, g, f0)
    iv0 = ref.eddy_interface_var(st) + 1e-30   # mode-projected metric (crux gate)
    print(f"[{name}] dt={st.dt:.3e} efold~{efold} budget={budget} shear={st._shear:.2f} "
          f"Ro1={ro1_0:.4f} Ro2={ro2_0:.4f} ivar0={iv0:.3e}", flush=True)

    best_ro1 = ro1_0
    best_ro2 = ro2_0
    best_ivar_x = 1.0
    reached_step = None
    blew_up = False
    saturated = False
    prev_ro2 = ro2_0
    steps = 0
    t0 = time.perf_counter()
    for chunk in range(0, budget, STEP_CHUNK):
        n = min(STEP_CHUNK, budget - chunk)
        try:
            for _ in range(n):
                ref.step_2layer(st)
        except (ValueError, AssertionError) as ex:
            print(f"[{name}] !! trap at ~step {steps + n}: {str(ex)[:60]}", flush=True)
            blew_up = True
            break
        steps += n
        ro1 = eddy_rossby(st.u1, st.v1, g, f0)
        ro2 = eddy_rossby(st.u2, st.v2, g, f0)
        if not (np.isfinite(ro1) and np.isfinite(ro2)):
            print(f"[{name}] !! non-finite at step {steps}", flush=True)
            blew_up = True
            break
        best_ro1 = max(best_ro1, ro1)
        best_ro2 = max(best_ro2, ro2)
        ivar = ref.eddy_interface_var(st) + 1e-30
        ivar_ratio = ivar / iv0          # mode amplitude^2 growth vs seed
        best_ivar_x = max(best_ivar_x, ivar_ratio)
        ms = (time.perf_counter() - t0) / steps * 1000.0
        print(f"[{name}] step={steps:6d} Ro1={ro1:.4f} Ro2={ro2:.4f} "
              f"ivar_x={ivar_ratio:.2e} {ms:.2f}ms/step", flush=True)
        if reached_step is None and max(ro1, ro2) >= RO_TARGET:
            reached_step = steps
            print(f"[{name}] *** Ro>={RO_TARGET} at step {steps} ***", flush=True)
        # Saturation: past target and lower-layer eddy Ro plateaued.
        if reached_step is not None and prev_ro2 > 1e-6 and abs(ro2 - prev_ro2) / prev_ro2 < 0.03:
            saturated = True
            print(f"[{name}] equilibrated at step {steps}", flush=True)
            break
        prev_ro2 = ro2

    # Reaching Ro>=0.10 then outcropping IS finite-amplitude saturation (a win),
    # so "reached" outranks "blew_up".
    if saturated:
        verdict = "SATURATED"
    elif reached_step:
        verdict = "REACHED+TRAP" if blew_up else "REACHED"
    elif blew_up:
        verdict = "BLEW_UP(no eddies)"
    else:
        verdict = "LAMINAR"
    line = (f"{name:16s} gp1={cfg[1]} xi={cfg[2]:<4} tau_rad={cfg[3]:<9.0f} tau_drag={cfg[4]:<9.0f} "
            f"sponge={cfg[5]} nu4={cfg[6]} | efold~{efold:<6d} steps={steps:<6d} "
            f"best_Ro1={best_ro1:.4f} best_Ro2={best_ro2:.4f} ivar_x={best_ivar_x:.2e} "
            f"reached@{reached_step} -> {verdict}")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print("\nRESULT: " + line, flush=True)


if __name__ == "__main__":
    run_config(int(sys.argv[1]))

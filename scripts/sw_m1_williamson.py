"""M1 Williamson-2 GPU balance report.

Runs the GPU Williamson-2 steady state (128x64, 80 steps) and prints
velocity_l2_drift, mass drift, energy drift, and potential-enstrophy drift.
Prints PASS/FAIL against the M1 gate thresholds.

Usage:
    uv run python scripts/sw_m1_williamson.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable (mirrors sw_spike_killgate.py pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.sim import sw_gpu  # noqa: E402
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402

# --- Configuration (matching the test) ---
W, H = 128, 64
A, OMEGA, U0, GP, H0 = 1.0, 2.0, 0.2, 1.0, 5.0
N_STEPS = 80

# --- Gate thresholds ---
GATE_VEL_L2  = 2e-2   # velocity_l2_drift < 2e-2
GATE_MASS_RT = 1e-5   # |mass_drift| / m0 < 1e-5


def main() -> None:
    print("=== M1 Williamson-2 GPU Balance Report ===")
    print(f"Grid: {W}x{H}  steps: {N_STEPS}  a={A}  omega={OMEGA}  u0={U0}  gp={GP}  h0={H0}")

    # Set up GPU context (mirrors sw_spike_killgate.py).
    gpu = GpuContext.headless()
    gpu.make_current()

    # Build solver from Williamson-2 initial condition.
    sg = sw_gpu.SwGpuSolver.from_williamson2(
        gpu, W=W, H=H, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0
    )

    # Capture initial diagnostics.
    m0 = sg.total_mass()

    # Also track energy and potential enstrophy using the CPU ref on the initial state.
    st0 = ref.williamson2_state(W=W, H=H, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0)
    e0  = ref.total_energy(st0)
    pe0 = ref.total_potential_enstrophy(st0)

    print(f"  m0       = {m0:.6e}")
    print(f"  e0 (CPU) = {e0:.6e}")
    print(f"  pe0(CPU) = {pe0:.6e}")

    # Run N_STEPS GPU steps.
    print(f"\nRunning {N_STEPS} GPU steps...")
    for _ in range(N_STEPS):
        sg.step()

    # Download state for diagnostics.
    hg, ug, vg = sg.download_state()

    # GPU diagnostics.
    vel_l2 = sg.velocity_l2_drift()
    m1     = sg.total_mass()
    mass_rtol = abs(m1 - m0) / abs(m0)

    # Energy and potential enstrophy via CPU ref on the GPU-downloaded state.
    import numpy as np
    import dataclasses

    st_gpu = dataclasses.replace(st0, h=hg.astype(float), u=ug.astype(float), v=vg.astype(float))
    e1  = ref.total_energy(st_gpu)
    pe1 = ref.total_potential_enstrophy(st_gpu)
    energy_rtol = abs(e1 - e0) / abs(e0)
    pe_rtol     = abs(pe1 - pe0) / abs(pe0)

    print(f"\n--- Results after {N_STEPS} steps ---")
    print(f"  velocity_l2_drift : {vel_l2:.6e}   (gate < {GATE_VEL_L2:.0e})")
    print(f"  mass drift (rtol) : {mass_rtol:.6e}   (gate < {GATE_MASS_RT:.0e})")
    print(f"  energy drift (rel): {energy_rtol:.6e}")
    print(f"  pot-enstrophy (rel): {pe_rtol:.6e}")

    vel_pass  = vel_l2 < GATE_VEL_L2
    mass_pass = mass_rtol < GATE_MASS_RT
    finite_ok = bool(np.all(np.isfinite(hg)))

    print(f"\n--- Gate evaluation ---")
    print(f"  finite h          : {'PASS' if finite_ok  else 'FAIL'}")
    print(f"  velocity_l2_drift : {'PASS' if vel_pass   else 'FAIL'}")
    print(f"  mass conservation : {'PASS' if mass_pass  else 'FAIL'}")

    overall = finite_ok and vel_pass and mass_pass
    print(f"\n{'PASS' if overall else 'FAIL'} — M1 Williamson-2 GPU balance gate")

    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    main()

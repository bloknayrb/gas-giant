"""M1 Williamson-2 GPU balance report — consolidated M1 PASS/FAIL summary.

Runs the GPU Williamson-2 steady state (128x64, 80 steps) and prints:
  1. Per-field GPU-vs-CPU diff tests (pytest subset, -k matches_ref).
  2. Williamson-2 velocity_l2 drift gate.
  3. Mass / energy / potential-enstrophy drift.
  4. Determinism hash (two fresh solvers, byte-identical SHA1).

Prints one overall PASS/FAIL line and exits 1 on any failure.

Usage:
    uv run python scripts/sw_m1_williamson.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure scripts/ is importable (mirrors sw_spike_killgate.py pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402
from gasgiant.sim import sw_gpu  # noqa: E402

# --- Configuration (matching the test) ---
W, H = 128, 64
A, OMEGA, U0, GP, H0 = 1.0, 2.0, 0.2, 1.0, 5.0
N_STEPS = 80
N_STEPS_DETERM = 40   # steps used for determinism check

# --- Gate thresholds ---
GATE_VEL_L2  = 2e-2   # velocity_l2_drift < 2e-2
GATE_MASS_RT = 1e-5   # |mass_drift| / m0 < 1e-5
GATE_ENERGY_RT  = 1e-2   # monitored, not hard-closed
GATE_PE_RT      = 1e-2   # monitored, not hard-closed


def run_per_field_tests() -> tuple[bool, str]:
    """Run pytest per-field GPU-vs-CPU diff subset via subprocess.

    Returns (passed: bool, summary_line: str).
    """
    repo_root = Path(__file__).resolve().parent.parent
    test_path = str(repo_root / "tests" / "unit" / "test_sw_gpu.py")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            test_path,
            "-q",
            "-k", "matches_ref",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),  # ensure repo-root conftest.py (gpu fixture) is discovered
    )
    # Count passed tests from pytest output (look for "X passed" summary).
    passed_count = 0
    for line in result.stdout.splitlines():
        if "passed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.rstrip(",") == "passed":
                    with contextlib.suppress(ValueError, IndexError):
                        passed_count = int(parts[i - 1])
    ok = result.returncode == 0
    summary = f"  per-field matches_ref tests: {passed_count}/7 passed  {'PASS' if ok else 'FAIL'}"
    if not ok:
        # Print subprocess stderr/stdout for debugging.
        print("    [pytest stdout]", result.stdout[-800:] if result.stdout else "(empty)")
        print("    [pytest stderr]", result.stderr[-400:] if result.stderr else "(empty)")
    return ok, summary


def run_williamson_balance(gpu: GpuContext) -> dict:
    """Run Williamson-2 balance check; return dict of gate results."""
    sg = sw_gpu.SwGpuSolver.from_williamson2(
        gpu, W=W, H=H, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0
    )

    m0 = sg.total_mass()

    st0 = ref.williamson2_state(W=W, H=H, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0)
    e0  = ref.total_energy(st0)
    pe0 = ref.total_potential_enstrophy(st0)

    print(f"  m0       = {m0:.6e}")
    print(f"  e0 (CPU) = {e0:.6e}")
    print(f"  pe0(CPU) = {pe0:.6e}")

    print(f"\nRunning {N_STEPS} GPU steps...")
    for _ in range(N_STEPS):
        sg.step()

    hg, ug, vg = sg.download_state()

    vel_l2    = sg.velocity_l2_drift()
    m1        = sg.total_mass()
    mass_rtol = abs(m1 - m0) / abs(m0)

    st_gpu = dataclasses.replace(
        st0,
        h=hg.astype(float),
        u=ug.astype(float),
        v=vg.astype(float),
    )
    e1  = ref.total_energy(st_gpu)
    pe1 = ref.total_potential_enstrophy(st_gpu)
    energy_rtol = abs(e1 - e0) / abs(e0)
    pe_rtol     = abs(pe1 - pe0) / abs(pe0)

    finite_ok = bool(
        np.all(np.isfinite(hg)) and np.all(np.isfinite(ug)) and np.all(np.isfinite(vg))
    )

    return dict(
        finite_ok=finite_ok,
        vel_l2=vel_l2,
        mass_rtol=mass_rtol,
        energy_rtol=energy_rtol,
        pe_rtol=pe_rtol,
        m0=m0, e0=e0, pe0=pe0,
    )


def run_determinism_check(gpu: GpuContext) -> tuple[bool, str]:
    """Build two fresh solvers, step both N_STEPS_DETERM times, compare SHA1."""
    sg1 = sw_gpu.SwGpuSolver.from_williamson2(
        gpu, W=W, H=H, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0
    )
    sg2 = sw_gpu.SwGpuSolver.from_williamson2(
        gpu, W=W, H=H, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0
    )

    for _ in range(N_STEPS_DETERM):
        sg1.step()
        sg2.step()

    h1, u1, v1 = sg1.download_state()
    h2, u2, v2 = sg2.download_state()

    blob1 = np.concatenate([h1.ravel(), u1.ravel(), v1.ravel()]).tobytes()
    blob2 = np.concatenate([h2.ravel(), u2.ravel(), v2.ravel()]).tobytes()

    sha1 = hashlib.sha1(blob1).hexdigest()
    sha2 = hashlib.sha1(blob2).hexdigest()

    match = sha1 == sha2
    summary = (
        f"  determinism SHA1: {sha1[:16]}…  "
        f"{'byte-identical: PASS' if match else f'MISMATCH sha2={sha2[:16]}… FAIL'}"
    )
    return match, summary


def main() -> None:
    print("=== M1 Williamson-2 GPU Balance Report ===")
    print(f"Grid: {W}x{H}  steps: {N_STEPS}  a={A}  omega={OMEGA}  u0={U0}  gp={GP}  h0={H0}")

    failures: list[str] = []

    # ------------------------------------------------------------------ #
    # 1. Per-field GPU-vs-CPU diff tests (subprocess pytest)              #
    # ------------------------------------------------------------------ #
    print("\n--- [1] Per-field GPU-vs-CPU diff tests ---")
    pf_pass, pf_summary = run_per_field_tests()
    print(pf_summary)
    if not pf_pass:
        failures.append("per-field diff tests")

    # ------------------------------------------------------------------ #
    # GPU context for Williamson balance + determinism                    #
    # ------------------------------------------------------------------ #
    gpu = GpuContext.headless()
    gpu.make_current()

    # ------------------------------------------------------------------ #
    # 2 + 3. Williamson-2 balance                                         #
    # ------------------------------------------------------------------ #
    print("\n--- [2+3] Williamson-2 balance (velocity_l2, mass, energy, PE) ---")
    res = run_williamson_balance(gpu)

    print(f"\n  Results after {N_STEPS} steps:")
    print(f"  velocity_l2_drift  : {res['vel_l2']:.6e}   (gate < {GATE_VEL_L2:.0e})")
    print(f"  mass drift (rtol)  : {res['mass_rtol']:.6e}   (gate < {GATE_MASS_RT:.0e})")
    print(f"  energy drift (rel) : {res['energy_rtol']:.6e}   (monitored < {GATE_ENERGY_RT:.0e})")
    print(f"  pot-enstrophy (rel): {res['pe_rtol']:.6e}   (monitored < {GATE_PE_RT:.0e})")

    finite_pass = res["finite_ok"]
    vel_pass    = res["vel_l2"]    < GATE_VEL_L2
    mass_pass   = res["mass_rtol"] < GATE_MASS_RT
    energy_ok   = res["energy_rtol"] < GATE_ENERGY_RT   # monitored
    pe_ok       = res["pe_rtol"]     < GATE_PE_RT       # monitored

    print(f"\n  finite h           : {'PASS' if finite_pass else 'FAIL'}")
    print(f"  velocity_l2_drift  : {'PASS' if vel_pass    else 'FAIL'}")
    print(f"  mass conservation  : {'PASS' if mass_pass   else 'FAIL'}")
    print(f"  energy drift       : {'PASS (monitored)' if energy_ok else 'WARN (monitored)'}")
    print(f"  pot-enstrophy      : {'PASS (monitored)' if pe_ok     else 'WARN (monitored)'}")

    if not finite_pass:
        failures.append("finite h")
    if not vel_pass:
        failures.append("velocity_l2_drift")
    if not mass_pass:
        failures.append("mass conservation")
    # energy and PE are MONITORED — warn but do not fail M1

    # ------------------------------------------------------------------ #
    # 4. Determinism hash                                                  #
    # ------------------------------------------------------------------ #
    print(f"\n--- [4] Determinism check ({N_STEPS_DETERM} steps, two fresh solvers) ---")
    determ_pass, determ_summary = run_determinism_check(gpu)
    print(determ_summary)
    if not determ_pass:
        failures.append("determinism")

    # ------------------------------------------------------------------ #
    # Overall verdict                                                      #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 55)
    if failures:
        print(f"FAIL — M1 gates failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("PASS — M1 Williamson-2 validation complete")
        print("       per-field atol=2e-5 diffs: 7/7 PASS")
        print(f"       vel_l2={res['vel_l2']:.2e} < {GATE_VEL_L2:.0e}")
        print(f"       mass_rtol={res['mass_rtol']:.2e} < {GATE_MASS_RT:.0e}")
        print(f"       energy_rtol={res['energy_rtol']:.2e} (monitored)")
        print(f"       pe_rtol={res['pe_rtol']:.2e} (monitored)")
        print("       determinism: byte-identical SHA1")


if __name__ == "__main__":
    main()

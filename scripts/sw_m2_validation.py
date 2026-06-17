"""M2 GPU semi-implicit solver — consolidated PASS/FAIL validation report.

Runs all M2 gates and prints a single PASS/FAIL summary line.
Exits non-zero on any failure.

Usage:
    uv run python scripts/sw_m2_validation.py
    python scripts/sw_m2_validation.py
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ensure the repo src is importable (mirrors sw_m1_williamson.py pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.sim import shallow_water_ref as ref  # noqa: E402
from gasgiant.sim import sw_gpu  # noqa: E402

# --- Configuration ---
W_W2, H_W2 = 64, 32          # Williamson-2 grid for SI tests (faster than 128x64)
A, OMEGA, U0, GP, H0 = 1.0, 2.0, 0.2, 1.0, 5.0
SI_THETA, SI_SOR, SI_HELMITERS, SI_PICARD = 0.5, 1.7, 200, 3

# --- Gate thresholds ---
GATE_ENERGY_RATIO_LO = 0.9
GATE_ENERGY_RATIO_HI = 1.1
GATE_VEL_L2 = 2e-2
GATE_MASS_RTOL = 1e-5
GATE_SOR50_ATOL = 5e-5   # SOR 50-iter match vs CPU ref


# ---------------------------------------------------------------------------
# (a) Per-field GPU↔CPU diff tests
# ---------------------------------------------------------------------------

def run_per_field_tests() -> tuple[bool, str]:
    """Run M2 per-field GPU-vs-CPU diff subset via subprocess pytest."""
    repo_root = Path(__file__).resolve().parent.parent
    test_path = str(repo_root / "tests" / "unit" / "test_m2_gpu.py")
    # Select all M2 GPU diff tests: helmholtz kernels + SI predictor + SI step + conservative cont.
    k_filter = (
        "matches_ref or converges_to_exact or a_scaling or smooth_tight"
        " or si_step_matches_ref or si_predictor_matches_ref"
    )
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            test_path,
            "-q",
            "-k", k_filter,
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )

    passed_count = 0
    failed_count = 0
    for line in result.stdout.splitlines():
        if "passed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.rstrip(",") == "passed":
                    with contextlib.suppress(ValueError, IndexError):
                        passed_count = int(parts[i - 1])
        if "failed" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.rstrip(",") == "failed":
                    with contextlib.suppress(ValueError, IndexError):
                        failed_count = int(parts[i - 1])

    ok = result.returncode == 0
    summary = (
        f"  per-field M2 diff tests: {passed_count} passed"
        + (f", {failed_count} failed" if failed_count else "")
        + f"  {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        print("    [pytest stdout]", result.stdout[-1200:] if result.stdout else "(empty)")
        print("    [pytest stderr]", result.stderr[-400:] if result.stderr else "(empty)")
    return ok, summary


# ---------------------------------------------------------------------------
# (b) GPU gravity-wave stability: resting layer + bump, dt = N * dt_gw
# ---------------------------------------------------------------------------

def run_gravity_wave_stability(gpu: GpuContext) -> tuple[bool, str, float, float]:
    """Semi-implicit resting-layer + bump: N=20x explicit gravity-wave dt.

    Returns (pass, summary, energy_ratio, N).
    """
    W, H, a, gp, h0 = 64, 32, 1.0, 1.0, 5.0
    omega = 0.0
    g = ref.Grid(W=W, H=H, a=a)

    lam = np.linspace(0.0, 2 * np.pi, W, endpoint=False)[None, :]
    phi = g.phi_c[:, None]
    bump = 0.1 * np.exp(-((phi) ** 2 + (lam - np.pi) ** 2) / 0.2)
    h = (h0 + bump).astype(np.float32)
    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H + 1, W), dtype=np.float32)

    c_gw = np.sqrt(gp * float(h.max()))
    cos_min = max(g.cos_c.min(), 1e-6)
    dx_min = min(cos_min * a * g.dlam, a * g.dphi)
    dt_gw = 0.3 * dx_min / c_gw
    N = 20
    dt_si = N * dt_gw

    sg = sw_gpu.SwGpuSolver(
        gpu, W=W, H=H, a=a, gp=gp, omega=omega, dt=dt_si, h_floor=0.05,
        semi_implicit=True, theta=SI_THETA, sor_omega=SI_SOR,
        helmholtz_iters=SI_HELMITERS, picard_iters=SI_PICARD, dt_multiplier=1.0,
    )
    sg._tex_h.write(h.tobytes())
    sg._tex_u.write(u.tobytes())
    sg._tex_v.write(v.tobytes())
    sg.u_init = u.copy()
    sg.v_init = v.copy()
    sg.H_ref = ref.reference_depth(h)

    e0 = sg.total_energy()
    n_steps = 40
    for _ in range(n_steps):
        sg.step()

    hh, uu, vv = sg.download_state()
    finite_ok = (
        bool(np.all(np.isfinite(hh)))
        and bool(np.all(np.isfinite(uu)))
        and bool(np.all(np.isfinite(vv)))
    )
    e1 = sg.total_energy()
    ratio = e1 / e0

    ok = finite_ok and (GATE_ENERGY_RATIO_LO <= ratio <= GATE_ENERGY_RATIO_HI)
    summary = (
        f"  gravity-wave stability: N={N}x dt_gw, {n_steps} steps, "
        f"energy_ratio={ratio:.4f} [{GATE_ENERGY_RATIO_LO},{GATE_ENERGY_RATIO_HI}]  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok, summary, ratio, float(N)


# ---------------------------------------------------------------------------
# (c) W2 geostrophic balance under SI step
# ---------------------------------------------------------------------------

def run_w2_geostrophic(gpu: GpuContext) -> tuple[bool, str, float, float]:
    """SI Williamson-2: check velocity_l2_drift + mass conservation."""
    sg = sw_gpu.SwGpuSolver.from_williamson2(
        gpu, W=W_W2, H=H_W2, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0,
        semi_implicit=True, theta=SI_THETA, sor_omega=SI_SOR,
        helmholtz_iters=SI_HELMITERS, picard_iters=SI_PICARD, dt_multiplier=1.0,
    )
    m0 = sg.total_mass()
    N_STEPS = 20
    for _ in range(N_STEPS):
        sg.step()

    vel_l2 = sg.velocity_l2_drift()
    m1 = sg.total_mass()
    mass_rtol = abs(m1 - m0) / abs(m0)

    vel_pass = vel_l2 < GATE_VEL_L2
    mass_pass = mass_rtol < GATE_MASS_RTOL
    ok = vel_pass and mass_pass
    summary = (
        f"  W2 geostrophic (SI, {N_STEPS} steps): "
        f"vel_l2={vel_l2:.2e} (gate<{GATE_VEL_L2:.0e}), "
        f"mass_rtol={mass_rtol:.2e} (gate<{GATE_MASS_RTOL:.0e})  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok, summary, vel_l2, mass_rtol


# ---------------------------------------------------------------------------
# (d) SI determinism: two runs, SHA1 match
# ---------------------------------------------------------------------------

def run_si_determinism(gpu: GpuContext) -> tuple[bool, str, str]:
    """Build two independent SI solvers from W2, step 5x, compare SHA1."""
    def _make_and_run():
        sg = sw_gpu.SwGpuSolver.from_williamson2(
            gpu, W=W_W2, H=H_W2, a=A, omega=OMEGA, u0=U0, gp=GP, h0=H0,
            semi_implicit=True, theta=SI_THETA, sor_omega=SI_SOR,
            helmholtz_iters=SI_HELMITERS, picard_iters=SI_PICARD, dt_multiplier=1.0,
        )
        for _ in range(5):
            sg.step()
        h, u, v = sg.download_state()
        m = hashlib.sha1()
        for arr in (h, u, v):
            m.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        return m.hexdigest()

    sha1 = _make_and_run()
    sha2 = _make_and_run()
    ok = sha1 == sha2
    summary = (
        f"  SI determinism SHA1: {sha1[:16]}…  "
        f"{'byte-identical: PASS' if ok else f'MISMATCH sha2={sha2[:16]}… FAIL'}"
    )
    return ok, summary, sha1


# ---------------------------------------------------------------------------
# (e) SOR-50 match gate (GPU vs CPU ref, atol=5e-5)
# ---------------------------------------------------------------------------

def run_sor50_match(gpu: GpuContext) -> tuple[bool, str, float]:
    """50-iter SOR GPU vs CPU ref — the tightest per-iter accumulation gate."""
    W, H = 64, 32
    g = ref.Grid(W=W, H=H, a=1.3)
    gp, theta, dt = 0.8, 0.5, 0.7
    rng = np.random.default_rng(11)
    rhs = rng.standard_normal((H, W)).astype(np.float32)
    H_ref_lat = (0.5 + rng.random(H)).astype(np.float32)
    n_iters, sor_omega = 50, 1.7

    cpu = ref.helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g, n_iters, sor_omega)
    out = sw_gpu.run_helmholtz_sor(gpu, rhs, H_ref_lat, gp, theta, dt, g.a, n_iters, sor_omega)
    max_diff = float(np.max(np.abs(out - cpu)))
    ok = max_diff <= GATE_SOR50_ATOL
    summary = (
        f"  SOR-50 GPU vs CPU max_diff={max_diff:.2e} (gate<={GATE_SOR50_ATOL:.0e})  "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok, summary, max_diff


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== M2 GPU Semi-Implicit Solver Validation Report ===")
    print(f"Grid W2: {W_W2}x{H_W2}  a={A}  omega={OMEGA}  u0={U0}  gp={GP}  h0={H0}")
    print(f"SI params: theta={SI_THETA}  sor_omega={SI_SOR}  "
          f"helmholtz_iters={SI_HELMITERS}  picard_iters={SI_PICARD}")

    failures: list[str] = []

    # ------------------------------------------------------------------ #
    # (a) Per-field M2 GPU↔CPU diff tests                                 #
    # ------------------------------------------------------------------ #
    print("\n--- [a] Per-field M2 GPU-vs-CPU diff tests (pytest subprocess) ---")
    pf_pass, pf_summary = run_per_field_tests()
    print(pf_summary)
    if not pf_pass:
        failures.append("per-field M2 diff tests")

    # GPU context for in-process gates.
    gpu = GpuContext.headless()
    gpu.make_current()

    # ------------------------------------------------------------------ #
    # (b) Gravity-wave stability                                           #
    # ------------------------------------------------------------------ #
    print("\n--- [b] GPU gravity-wave stability (resting layer + bump, N=20x dt_gw) ---")
    gw_pass, gw_summary, gw_ratio, gw_N = run_gravity_wave_stability(gpu)
    print(gw_summary)
    if not gw_pass:
        failures.append("gravity-wave stability")

    # ------------------------------------------------------------------ #
    # (c) W2 geostrophic balance under SI                                  #
    # ------------------------------------------------------------------ #
    print("\n--- [c] Williamson-2 geostrophic balance (SI, 20 steps) ---")
    w2_pass, w2_summary, w2_vel, w2_mass = run_w2_geostrophic(gpu)
    print(w2_summary)
    if not w2_pass:
        failures.append("W2 geostrophic SI balance")

    # ------------------------------------------------------------------ #
    # (d) SI determinism                                                   #
    # ------------------------------------------------------------------ #
    print("\n--- [d] SI determinism (two fresh runs, SHA1) ---")
    det_pass, det_summary, sha1 = run_si_determinism(gpu)
    print(det_summary)
    if not det_pass:
        failures.append("SI determinism")

    # ------------------------------------------------------------------ #
    # (e) SOR-50 match gate                                                #
    # ------------------------------------------------------------------ #
    print("\n--- [e] SOR-50 GPU vs CPU ref (atol gate) ---")
    sor_pass, sor_summary, sor_diff = run_sor50_match(gpu)
    print(sor_summary)
    if not sor_pass:
        failures.append("SOR-50 match")

    # ------------------------------------------------------------------ #
    # Overall verdict                                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    if failures:
        print(f"FAIL — M2 gates failed: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("PASS — M2 semi-implicit validation complete")
        print("       per-field M2 diff tests: all PASS")
        print(f"       gravity-wave N={int(gw_N)}x dt_gw stability: "
              f"energy_ratio={gw_ratio:.4f}")
        print(f"       W2 SI geostrophic: vel_l2={w2_vel:.2e}, "
              f"mass_rtol={w2_mass:.2e}")
        print(f"       SI determinism: byte-identical SHA1 {sha1[:16]}…")
        print(f"       SOR-50 GPU vs CPU: max_diff={sor_diff:.2e} "
              f"(<= {GATE_SOR50_ATOL:.0e})")


if __name__ == "__main__":
    main()

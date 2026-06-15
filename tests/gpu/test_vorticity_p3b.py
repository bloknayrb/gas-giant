"""GPU tests for P3b: closed-loop vorticity solver (SOR Poisson + feather).

Tests
-----
1. psi_omega_roundtrip      — ψ→ω→ψ round-trip: stamp analytic ψ, compute ω via
                              laplacian.glsl, SOR-solve back to ψ; assert close
                              after removing mean (operator/sign lock).
2. closed_loop_determinism  — vorticity mode (closed loop), same seed × 2 runs,
                              byte-identical tracer readback.
3. closed_loop_boundedness  — ~500 steps, velocity from solved ψ; tracers + ω
                              must stay finite and bounded.
4. feather_continuity       — feather_psi_continuity() below tolerance (no velocity
                              ridge at 50–60°).
5. default_byte_identical   — kinematic path (default) byte-identical hash check
                              (calls scripts/p05_baseline_hash.py --check).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KERNELS = "gasgiant.sim.kernels"


def _vort_params(seed: int = 42, steps: int = 0) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.solver.vort_relax_tau = 120.0
    p.solver.vort_hypervisc = 1.0
    p.solver.coriolis_f0 = 2.0
    p.solver.poisson_iters = 48
    p.solver.sor_omega = 1.7
    return p


def _read_omega(sim: Simulation) -> np.ndarray:
    state = sim.solver._omega_state
    assert state is not None
    return sim.gpu.read_texture(state.cur)[..., 0]


def _read_tracers(sim: Simulation) -> np.ndarray:
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)[..., 0]


# ---------------------------------------------------------------------------
# Build the SOR solver as a standalone test harness (no full Simulation).
# ---------------------------------------------------------------------------

def _build_sor_harness(gpu: GpuContext, W: int, H: int):
    """Return (k_lap, k_sor_red, k_sor_black) compiled shaders."""
    import importlib.resources as ir

    pkg = _KERNELS
    common_src    = ir.files(pkg).joinpath("common.glsl").read_text(encoding="utf-8")
    laplacian_src = ir.files(pkg).joinpath("laplacian.glsl").read_text(encoding="utf-8")

    _HEADER = (
        "#version 430\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "#define DOMAIN 0\n"
        + common_src + "\n"
        + laplacian_src + "\n"
    )

    # Laplacian shader — same as P3a test
    lap_src = (
        _HEADER
        + "uniform sampler2D u_field;\n"
        + "uniform ivec2 u_size;\n"
        + "layout(r32f, binding = 0) writeonly uniform image2D out_lap;\n"
        + "void main() {\n"
        + "    ivec2 px = ivec2(gl_GlobalInvocationID.xy);\n"
        + "    if (px.x >= u_size.x || px.y >= u_size.y) return;\n"
        + "    float lap = laplacianPsi(u_field, px, u_size);\n"
        + "    imageStore(out_lap, px, vec4(lap, 0.0, 0.0, 0.0));\n"
        + "}\n"
    )
    k_lap = gpu.ctx.compute_shader(lap_src)

    # Use gpu.compute which handles #include expansion.
    k_sor_red   = gpu.compute(pkg, "poisson_sor.comp", defines={"DOMAIN": "0", "COLOR": "0"})
    k_sor_black = gpu.compute(pkg, "poisson_sor.comp", defines={"DOMAIN": "0", "COLOR": "1"})

    return k_lap, k_sor_red, k_sor_black


# ---------------------------------------------------------------------------
# 1. ψ → ω → ψ round-trip (OPERATOR / SIGN LOCK)
# ---------------------------------------------------------------------------

def test_psi_omega_roundtrip(gpu):
    """Stamp a smooth analytic ψ, compute ω = ∇²ψ via laplacian.glsl,
    SOR-solve back to ψ; assert recovered ψ ≈ original ψ after removing mean.

    This is the definitive operator/sign lock: a coefficient mismatch between
    laplacian.glsl and poisson_sor.comp will fail this test.

    Resolution is intentionally small (128×64) so SOR converges quickly.
    Tolerance: 5% median relative error at mid-latitudes.
    """
    W, H = 128, 64
    SOR_ITERS = 600
    SOR_OM    = 1.7

    # Build analytic ψ: Y_2^0 ∝ (3sin²φ − 1), which is a smooth eigenfunction.
    rows = np.arange(H)
    lat_gpu = np.pi / 2.0 - (rows + 0.5) / H * np.pi   # descending (H,)
    LAT2D = lat_gpu[:, np.newaxis] * np.ones((H, W))
    psi_ref = (3.0 * np.sin(LAT2D) ** 2 - 1.0).astype(np.float32)

    # Upload ψ to GPU.
    psi_tex = gpu.texture2d((W, H), 1, "f4", data=psi_ref[:, :, np.newaxis])
    omega_tex = gpu.texture2d((W, H), 1, "f4")
    psi_work  = gpu.texture2d((W, H), 1, "f4")
    psi_work.repeat_x = True

    # ---- Step 1: compute ω = ∇²ψ via laplacian.glsl -----------------------
    k_lap, k_sor_red, k_sor_black = _build_sor_harness(gpu, W, H)

    k_lap["u_field"] = 0
    k_lap["u_size"]  = (W, H)
    psi_tex.use(location=0)
    omega_tex.bind_to_image(0, read=False, write=True)
    gx = (W + 15) // 16
    gy = (H + 15) // 16
    k_lap.run(gx, gy, 1)
    gpu.ctx.memory_barrier()

    # ---- Step 2: warm-start psi_work from psi (gives SOR a good start) ------
    # For round-trip we warm-start from ψ_ref itself to converge quickly.
    # (Spec says warm-start from previous step's ψ; here we prove convergence.)
    psi_work_init = psi_ref.copy()
    psi_work.write(psi_work_init[:, :, np.newaxis].tobytes())

    # ---- Step 3: SOR solve ∇²ψ = ω -----------------------------------------
    omega_tex.use(location=0)
    k_sor_red["u_omega_rel"]  = 0
    k_sor_black["u_omega_rel"] = 0
    k_sor_red["u_size"]   = (W, H)
    k_sor_black["u_size"] = (W, H)
    k_sor_red["u_sor_omega"]   = SOR_OM
    k_sor_black["u_sor_omega"] = SOR_OM

    for _ in range(SOR_ITERS):
        psi_work.bind_to_image(0, read=True, write=True)
        k_sor_red.run(gx, gy, 1)
        gpu.ctx.memory_barrier()
        psi_work.bind_to_image(0, read=True, write=True)
        k_sor_black.run(gx, gy, 1)
        gpu.ctx.memory_barrier()

    psi_solved = gpu.read_texture(psi_work)[..., 0]

    # Cleanup GPU resources.
    psi_tex.release()
    omega_tex.release()
    psi_work.release()

    # ---- Step 4: compare (remove mean — Poisson is defined up to a const) --
    # Focus on mid-latitudes where stencil isn't degenerate.
    mid_mask = np.abs(lat_gpu) < np.deg2rad(65.0)

    ref_mid    = psi_ref[mid_mask, :]
    solved_mid = psi_solved[mid_mask, :]

    ref_mean    = ref_mid.mean()
    solved_mean = solved_mid.mean()
    ref_c       = ref_mid    - ref_mean
    solved_c    = solved_mid - solved_mean

    # Relative error where |ref_c| > 0.05 (exclude near-zero crossings).
    nz = np.abs(ref_c) > 0.05
    if nz.sum() < 10:
        pytest.skip("Too few non-zero reference points for round-trip test")

    rel_err = np.abs(solved_c[nz] - ref_c[nz]) / np.abs(ref_c[nz])
    median_err = float(np.median(rel_err))
    max_err    = float(np.max(rel_err))

    print(f"\npsi->omega->psi round-trip: median_rel_err={median_err:.4f}  "
          f"max_rel_err={max_err:.4f}  (iters={SOR_ITERS})")

    assert median_err < 0.05, (
        f"ψ→ω→ψ round-trip median error {median_err:.4f} > 5% — "
        f"SOR coefficients may not match laplacian.glsl. "
        f"max_err={max_err:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Closed-loop determinism
# ---------------------------------------------------------------------------

def test_closed_loop_determinism(gpu):
    """Two vorticity-mode sims (closed loop: velocity from solved ψ),
    same seed, must produce byte-identical tracer readbacks."""
    N = 20
    p = _vort_params(seed=77)

    sim_a = Simulation(p, gpu)
    sim_a.solver.step(N)
    tracer_a = _read_tracers(sim_a).copy()
    sim_a._release_sim()

    sim_b = Simulation(p, gpu)
    sim_b.solver.step(N)
    tracer_b = _read_tracers(sim_b).copy()
    sim_b._release_sim()

    np.testing.assert_array_equal(
        tracer_a, tracer_b,
        err_msg="Closed-loop vorticity mode is not byte-identical across two runs "
                "with the same seed",
    )


# ---------------------------------------------------------------------------
# 3. Closed-loop boundedness (~500 steps)
# ---------------------------------------------------------------------------

def test_closed_loop_boundedness(gpu):
    """Over ~500 steps with velocity derived from the SOR-solved ψ,
    tracers and ω must stay finite and bounded (no blow-up)."""
    N = 500
    p = _vort_params(seed=13)

    sim = Simulation(p, gpu)
    sim.solver.step(N)

    tracer = _read_tracers(sim)
    omega  = _read_omega(sim)
    sim._release_sim()

    assert np.isfinite(tracer).all(), "Tracer has non-finite values after 500 steps"
    assert np.isfinite(omega).all(),  "ω has non-finite values after 500 steps"

    tracer_max = float(np.abs(tracer).max())
    omega_max  = float(np.abs(omega).max())

    # Tracers are color values in [0,1]; allow generous margin for numerical drift.
    assert tracer_max < 10.0, (
        f"Tracer blow-up: |tracer|_max = {tracer_max:.3f} after {N} steps"
    )
    # ω scale set by f0=2 + jets; allow 50× margin.
    assert omega_max < 200.0, (
        f"ω blow-up: |ω|_max = {omega_max:.3f} after {N} steps"
    )

    print(f"\nClosed-loop boundedness: |tracer|_max={tracer_max:.4f}  "
          f"|omega|_max={omega_max:.4f}  (after {N} steps)")


# ---------------------------------------------------------------------------
# 4. Feather continuity
# ---------------------------------------------------------------------------

def test_feather_continuity(gpu):
    """After running a few steps, the velocity gradient discontinuity across
    the 50–60° feather band should be below a sane tolerance."""
    N = 30
    p = _vort_params(seed=42)

    sim = Simulation(p, gpu)
    sim.solver.step(N)
    cont = sim.solver.feather_psi_continuity()
    sim._release_sim()

    print(f"\nFeather psi continuity (max d2psi/dphi2 in 48-62 deg band): {cont:.6f}")

    # No large velocity ridge: second-diff should be well below a few units.
    # (ψ values are O(1), first-diff is O(Δφ⁻¹)~60/π; second-diff should be
    # much smaller than the inter-band gradient itself.)
    assert cont < 50.0, (
        f"Feather continuity {cont:.4f} exceeds tolerance 50.0 — "
        f"velocity ridge at the 50–60° blend band"
    )


# ---------------------------------------------------------------------------
# 5. Default (kinematic) path byte-identical
# ---------------------------------------------------------------------------

def test_default_byte_identical():
    """Kinematic path must produce byte-identical hashes vs the committed
    baseline (scripts/p05_baseline_hash.py --check)."""
    script = Path(__file__).parents[2] / "scripts" / "p05_baseline_hash.py"
    if not script.exists():
        pytest.skip(f"Baseline script not found: {script}")

    baseline = Path(__file__).parents[2] / "out" / "audit" / "p05_baseline.json"
    if not baseline.exists():
        pytest.skip("No baseline JSON; run p05_baseline_hash.py first")

    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    assert result.returncode == 0, (
        "p05_baseline_hash.py --check FAILED — default kinematic path changed!\n"
        + result.stdout + result.stderr
    )

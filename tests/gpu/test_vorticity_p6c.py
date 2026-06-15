"""GPU tests for P6c: patch omega solver (full 3-domain vorticity).

Tests
-----
1. patch_omega_evolves          -- patch omega changes non-trivially between step N and 2N.
2. patch_omega_responds_to_param -- two runs differing only in vort_hypervisc produce
                                   DIFFERENT patch omega (not byte-identical).
3. patch_omega_bounded          -- ~400 steps at res 512; patch omega finite, |omega| < ceiling.
4. patch_psi_omega_roundtrip    -- stamp smooth psi on a patch, omega=laplacianPsi, patch SOR to
                                   convergence, recovered psi ~= original mid-cap within 5%.
5. full_vorticity_render_clean  -- render jupiter_like+vorticity; finite, no NaN, clip ~0.
6. patch_determinism            -- same seed -> patch omega byte-identical (assert_array_equal).
"""

from __future__ import annotations

import importlib.resources as ir

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.solver import DOMAIN_NORTH, DOMAIN_SOUTH, RHO_MAX

pytestmark = pytest.mark.gpu

_KERNELS = "gasgiant.sim.kernels"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vort_params(seed: int = 42, steps: int = 0, hypervisc: float = 1.0) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512   # minimum allowed; patches are ~59px at 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.solver.vort_relax_tau = 120.0
    p.solver.vort_hypervisc = hypervisc
    p.solver.coriolis_f0 = 2.0
    p.solver.poisson_iters = 32
    p.solver.sor_omega = 1.7
    return p


def _read_patch_omega(sim: Simulation, domain_kind: int) -> np.ndarray:
    """Read the current omega texture for the given patch domain."""
    state = sim.solver._omega_states[domain_kind]
    assert state is not None, f"No omega state for domain {domain_kind}"
    return sim.gpu.read_texture(state.cur)[..., 0]


def _patch_cap_mask(arr: np.ndarray, rho_max: float) -> np.ndarray:
    """Boolean mask for pixels where |lat| > 67deg (deep in the patch cap)."""
    N = arr.shape[0]
    idx = (np.arange(N) + 0.5) / N * 2.0 - 1.0
    s = (idx * rho_max)[np.newaxis, :] * np.ones((N, 1))
    t = (idx * rho_max)[:, np.newaxis] * np.ones((1, N))
    rho = np.hypot(s, t)
    # |lat| > 67deg <=> rho < 23deg (for north patch: lat = 90deg - rho_deg)
    return rho < np.deg2rad(23.0)


# ---------------------------------------------------------------------------
# Test 1: Patch omega EVOLVES
# ---------------------------------------------------------------------------

def test_patch_omega_evolves(gpu):
    """In vorticity mode, patch omega at step N differs non-trivially from step 2N.

    If the patch kernels are no-ops (e.g. u_size=(0,0) bug), the omega field is
    static and this test fails.
    """
    N = 30
    p = _vort_params(seed=11, steps=0)

    sim = Simulation(p, gpu)
    sim.solver.step(N)
    omega_n = _read_patch_omega(sim, DOMAIN_NORTH).copy()

    sim.solver.step(N)  # another N steps = 2N total
    omega_2n = _read_patch_omega(sim, DOMAIN_NORTH).copy()
    sim._release_sim()

    mask = _patch_cap_mask(omega_n, RHO_MAX)

    diff = np.abs(omega_2n[mask] - omega_n[mask])
    max_diff = float(diff.max())

    print(f"\npatch omega evolves: max_diff in cap = {max_diff:.4f}")

    assert max_diff > 1e-4, (
        f"Patch omega did not evolve between step {N} and {2*N}: max_diff={max_diff:.2e}. "
        "This likely means the patch omega kernels are no-ops (u_size not set, or "
        "DOMAIN define not passed to the kernel)."
    )


# ---------------------------------------------------------------------------
# Test 2: Patch omega RESPONDS to a param
# ---------------------------------------------------------------------------

def test_patch_omega_responds_to_param(gpu):
    """Two runs differing only in vort_hypervisc produce DIFFERENT patch omega.

    If patch kernel uniforms are not wired (the no-op bug), the two runs
    will be byte-identical regardless of hypervisc.
    """
    N = 50
    seed = 99

    sim_a = Simulation(_vort_params(seed=seed, hypervisc=0.5), gpu)
    sim_a.solver.step(N)
    omega_a = _read_patch_omega(sim_a, DOMAIN_NORTH).copy()
    sim_a._release_sim()

    sim_b = Simulation(_vort_params(seed=seed, hypervisc=2.0), gpu)
    sim_b.solver.step(N)
    omega_b = _read_patch_omega(sim_b, DOMAIN_NORTH).copy()
    sim_b._release_sim()

    # They must NOT be byte-identical.
    assert not np.array_equal(omega_a, omega_b), (
        "Patch omega is byte-identical for hypervisc=0.5 and hypervisc=2.0 after "
        f"{N} steps -- patch kernels may be ignoring u_hypervisc (no-op bug)."
    )

    # Quantify: they should differ by more than floating-point noise.
    diff = np.abs(omega_a - omega_b)
    max_diff = float(diff.max())
    print(f"\nhypervisc param response: max |omega_a - omega_b| = {max_diff:.4f}")
    assert max_diff > 1e-3, (
        f"Patch omega differs by only {max_diff:.2e} between hypervisc 0.5 and 2.0 -- "
        "difference is too small to confirm param wiring."
    )


# ---------------------------------------------------------------------------
# Test 3: Patch omega BOUNDED
# ---------------------------------------------------------------------------

def test_patch_omega_bounded(gpu):
    """~400 steps at res 512; patch omega must stay finite and |omega| < ceiling.

    OMEGA_CEILING is 60.0 (from omega_force.comp). Steady state is ~O(10).
    """
    STEPS = 400
    OMEGA_CEILING = 60.0

    p = _vort_params(seed=7, steps=0)
    p.sim.resolution = 512

    sim = Simulation(p, gpu)
    sim.solver.step(STEPS)

    for kind in (DOMAIN_NORTH, DOMAIN_SOUTH):
        omega = _read_patch_omega(sim, kind)
        name = "NORTH" if kind == DOMAIN_NORTH else "SOUTH"

        finite_ok = bool(np.all(np.isfinite(omega)))
        max_abs = float(np.max(np.abs(omega)))
        print(f"\n{name} patch after {STEPS} steps: finite={finite_ok}  |omega|_max={max_abs:.3f}")

        assert finite_ok, f"Patch {name} omega contains NaN/Inf after {STEPS} steps"
        assert max_abs < OMEGA_CEILING, (
            f"Patch {name} omega blew up: |omega|_max={max_abs:.2f} >= ceiling {OMEGA_CEILING}"
        )

    sim._release_sim()


# ---------------------------------------------------------------------------
# Test 4: Patch psi->omega->psi round-trip
# ---------------------------------------------------------------------------

def test_patch_psi_omega_roundtrip(gpu):
    """Stamp a smooth analytic psi on a patch, compute omega=nabla2psi (AE Laplacian),
    SOR-solve nabla2psi = omega back to psi; assert recovered psi ~= original in mid-cap
    (excluding apron and pole pixel) within 5% median relative error.

    This locks the AE SOR coefficients against the AE Laplacian -- a coefficient
    mismatch causes a large round-trip error.
    """
    N = 128   # patch NxN; small enough for SOR to converge quickly
    SOR_ITERS = 800
    SOR_OM = 1.7

    rho_max = RHO_MAX

    # Build Y_2^0 = 3cos2rho - 1 on the patch grid (smooth, non-trivial).
    idx = (np.arange(N) + 0.5) / N * 2.0 - 1.0
    coords = idx * rho_max
    ss, tt = np.meshgrid(coords, coords)
    rho2d = np.hypot(ss, tt)
    psi_ref = (3.0 * np.cos(rho2d) ** 2 - 1.0).astype(np.float32)

    # Upload psi to GPU (R32F).
    psi_tex   = gpu.texture2d((N, N), 1, "f4", data=psi_ref[:, :, np.newaxis])
    omega_tex = gpu.texture2d((N, N), 1, "f4")
    psi_work  = gpu.texture2d((N, N), 1, "f4")
    psi_work.repeat_x = False  # patch: clamp both axes

    # Build the patch lap and SOR kernels (DOMAIN=1).
    pkg = _KERNELS
    common_src    = ir.files(pkg).joinpath("common.glsl").read_text(encoding="utf-8")
    laplacian_src = ir.files(pkg).joinpath("laplacian.glsl").read_text(encoding="utf-8")

    _HEADER = (
        "#version 430\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "#define DOMAIN 1\n"
        + common_src + "\n"
        + laplacian_src + "\n"
    )

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
    k_lap["u_rho_max"] = rho_max

    k_sor_red   = gpu.compute(pkg, "poisson_sor.comp", defines={"DOMAIN": "1", "COLOR": "0"})
    k_sor_black = gpu.compute(pkg, "poisson_sor.comp", defines={"DOMAIN": "1", "COLOR": "1"})
    for k in (k_sor_red, k_sor_black):
        k["u_rho_max"] = rho_max

    gx = (N + 15) // 16
    gy = (N + 15) // 16

    # Step 1: omega = nabla2psi via AE Laplacian.
    k_lap["u_field"] = 0
    k_lap["u_size"]  = (N, N)
    psi_tex.use(location=0)
    omega_tex.bind_to_image(0, read=False, write=True)
    k_lap.run(gx, gy, 1)
    gpu.ctx.memory_barrier()

    # Step 2: warm-start psi_work from psi_ref.
    psi_work.write(psi_ref[:, :, np.newaxis].tobytes())

    # Step 3: SOR solve nabla2psi = omega.
    omega_tex.use(location=0)
    for k in (k_sor_red, k_sor_black):
        k["u_omega_rel"] = 0
        k["u_size"]      = (N, N)
        k["u_sor_omega"] = SOR_OM

    for _ in range(SOR_ITERS):
        psi_work.bind_to_image(0, read=True, write=True)
        k_sor_red.run(gx, gy, 1)
        gpu.ctx.memory_barrier()
        psi_work.bind_to_image(0, read=True, write=True)
        k_sor_black.run(gx, gy, 1)
        gpu.ctx.memory_barrier()

    psi_solved = gpu.read_texture(psi_work)[..., 0]

    psi_tex.release()
    omega_tex.release()
    psi_work.release()

    # Step 4: compare in mid-cap (rho in [3deg, 25deg]).
    interior = (rho2d > np.deg2rad(3.0)) & (rho2d < np.deg2rad(25.0))

    ref_int    = psi_ref[interior]
    solved_int = psi_solved[interior]

    # Remove mean (Poisson defined up to additive constant).
    ref_c    = ref_int    - ref_int.mean()
    solved_c = solved_int - solved_int.mean()

    nz = np.abs(ref_c) > 0.05
    if nz.sum() < 10:
        pytest.skip("Too few non-zero reference points in mid-cap")

    rel_err = np.abs(solved_c[nz] - ref_c[nz]) / np.abs(ref_c[nz])
    median_err = float(np.median(rel_err))
    max_err    = float(np.max(rel_err))

    print(
        f"\nPatch psi->omega->psi round-trip: median_err={median_err:.4f}  "
        f"max_err={max_err:.4f}  (iters={SOR_ITERS})"
    )

    assert median_err < 0.05, (
        f"Patch psi->omega->psi round-trip median error {median_err:.4f} > 5% -- "
        f"AE SOR coefficients may not match laplacian.glsl AE branch. "
        f"max_err={max_err:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: Full 3-domain vorticity render is CLEAN
# ---------------------------------------------------------------------------

def test_full_vorticity_render_clean(gpu):
    """Render jupiter_like + vorticity mode at modest res; assert finite, no NaN."""
    p = load_factory_preset("jupiter_like")
    p.sim.resolution = 512
    p.sim.dev_steps  = 40
    p.solver.type = SolverType.VORTICITY
    p.solver.poisson_iters = 24
    p.solver.sor_omega = 1.7
    p.solver.coriolis_f0 = 2.0
    p.solver.vort_relax_tau = 120.0
    p.solver.vort_hypervisc = 1.0

    sim = Simulation(p, gpu)
    maps = sim.render_maps(512)
    sim._release_sim()

    for name, arr in maps.items():
        assert np.all(np.isfinite(arr)), f"render map '{name}' contains NaN/Inf"
        clipped = np.mean((arr < 0.0) | (arr > 1.0))
        assert clipped < 0.02, (
            f"render map '{name}' has {clipped:.1%} pixels outside [0,1] -- "
            "likely a NaN or blow-up in vorticity mode"
        )
        print(f"  {name}: finite=True  clip_frac={clipped:.4f}")


# ---------------------------------------------------------------------------
# Test 6: Determinism
# ---------------------------------------------------------------------------

def test_patch_determinism(gpu):
    """Vorticity mode, same seed; patch omega byte-identical across 2 runs."""
    N = 25
    p = _vort_params(seed=55, steps=0)

    sim_a = Simulation(p, gpu)
    sim_a.solver.step(N)
    omega_a_n = _read_patch_omega(sim_a, DOMAIN_NORTH).copy()
    omega_a_s = _read_patch_omega(sim_a, DOMAIN_SOUTH).copy()
    sim_a._release_sim()

    sim_b = Simulation(p, gpu)
    sim_b.solver.step(N)
    omega_b_n = _read_patch_omega(sim_b, DOMAIN_NORTH).copy()
    omega_b_s = _read_patch_omega(sim_b, DOMAIN_SOUTH).copy()
    sim_b._release_sim()

    np.testing.assert_array_equal(
        omega_a_n, omega_b_n,
        err_msg="Patch NORTH omega is not byte-identical across two runs with the same seed",
    )
    np.testing.assert_array_equal(
        omega_a_s, omega_b_s,
        err_msg="Patch SOUTH omega is not byte-identical across two runs with the same seed",
    )

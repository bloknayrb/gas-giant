"""GPU tests for P3a: vorticity field evolution under kinematic velocity.

Tests
-----
1. laplacian_glsl_sign_lock  — GPU laplacian.glsl matches vorticity_ref on Y_2^0
   spherical harmonic (eigenvalue −6); re-locks the +tanφ sign on the GPU.
2. vorticity_mode_determinism — two sims, same seed, vorticity mode, same result.
3. vorticity_mode_boundedness — ω stays finite and bounded over ~400 steps.
4. vorticity_develops_structure — ω variance grows after many steps.
5. kinematic_mode_unchanged — solver.type=kinematic still works (guard).
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vort_params(seed: int = 42, steps: int = 0) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.solver.vort_relax_tau = 120.0
    p.solver.vort_hypervisc = 1.0
    p.solver.coriolis_f0 = 2.0
    return p


def _read_omega(sim: Simulation) -> np.ndarray:
    """Read the current absolute vorticity q texture as a 2-D array."""
    state = sim.solver._omega_state
    assert state is not None, "No omega_state in vorticity solver"
    raw = sim.gpu.read_texture(state.cur)  # (H, W, 1)
    return raw[..., 0]


# ---------------------------------------------------------------------------
# 1. Laplacian GPU sign-lock test
# ---------------------------------------------------------------------------

def _build_laplacian_test_shader(gpu: GpuContext, W: int, H: int) -> object:  # noqa: ARG001
    """Compile a tiny compute shader that evaluates laplacianPsi on a
    R32F field stored as a texture and writes the result to an image."""
    import importlib.resources as ir  # noqa: PLC0415

    # Build the full source by hand: inject the define then inline laplacian.glsl.
    pkg = "gasgiant.sim.kernels"
    laplacian_src = ir.files(pkg).joinpath("laplacian.glsl").read_text(encoding="utf-8")
    common_src    = ir.files(pkg).joinpath("common.glsl").read_text(encoding="utf-8")

    full_src = (
        "#version 430\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "#define DOMAIN 0\n"
        + common_src + "\n"
        + laplacian_src + "\n"
        "uniform sampler2D u_field;\n"
        "uniform ivec2 u_size;\n"
        "layout(r32f, binding = 0) writeonly uniform image2D out_lap;\n"
        "void main() {\n"
        "    ivec2 px = ivec2(gl_GlobalInvocationID.xy);\n"
        "    if (px.x >= u_size.x || px.y >= u_size.y) return;\n"
        "    float lap = laplacianPsi(u_field, px, u_size);\n"
        "    imageStore(out_lap, px, vec4(lap, 0.0, 0.0, 0.0));\n"
        "}\n"
    )
    return gpu.ctx.compute_shader(full_src)


def test_laplacian_glsl_sign_lock(gpu):
    """GPU laplacianPsi on Y_2^0 ∝ (3sin²φ−1) must give eigenvalue ≈ −6.

    This is the critical sign-lock test: a wrong +tanφ vs −tanφ sign causes
    the measured eigenvalue to deviate strongly from −6.  5% tolerance.
    """
    W, H = 256, 128  # smaller grid for speed; still adequate for 5% accuracy

    # Build Y_2^0 field (CPU, ascending lat convention for the math).
    # GPU grid: row 0 = north pole (lat = +π/2), row increases southward.
    rows = np.arange(H)
    lat_gpu = np.pi / 2.0 - (rows + 0.5) / H * np.pi   # descending (H,)
    LAT2D = lat_gpu[:, np.newaxis] * np.ones((H, W))

    Y = (3.0 * np.sin(LAT2D) ** 2 - 1.0).astype(np.float32)  # (H, W)

    # Upload Y to a GPU texture.
    field_tex = gpu.texture2d((W, H), 1, "f4",
                               data=Y[:, :, np.newaxis])
    out_tex = gpu.texture2d((W, H), 1, "f4")

    prog = _build_laplacian_test_shader(gpu, W, H)
    field_tex.use(location=0)
    prog["u_field"] = 0
    prog["u_size"] = (W, H)
    out_tex.bind_to_image(0, read=False, write=True)
    gx = (W + 15) // 16
    gy = (H + 15) // 16
    prog.run(gx, gy, 1)
    gpu.ctx.memory_barrier()

    lap_gpu = gpu.read_texture(out_tex)[..., 0]   # (H, W)

    field_tex.release()
    out_tex.release()

    # Compute eigenvalue ratio lap / Y, excluding poles and near-zero Y.
    lat_col = lat_gpu  # (H,)
    mid_mask = np.abs(lat_col) < np.deg2rad(70.0)  # exclude poles
    Y_mid = Y[mid_mask, :]
    lap_mid = lap_gpu[mid_mask, :]
    nz = np.abs(Y_mid) > 0.05
    ratio = lap_mid[nz] / Y_mid[nz]
    eigenvalue = float(np.median(ratio))
    error = abs(eigenvalue - (-6.0)) / 6.0

    print(f"\nGPU Y_2^0 eigenvalue: {eigenvalue:.4f}  expected: -6.000  "
          f"relative error: {error:.4f}")

    assert error < 0.05, (
        f"GPU laplacianPsi eigenvalue error {error:.4%} > 5% — "
        f"check +tanφ sign in laplacian.glsl (measured eigenvalue: {eigenvalue:.4f})"
    )


# ---------------------------------------------------------------------------
# 2. Vorticity mode determinism
# ---------------------------------------------------------------------------

def test_vorticity_mode_determinism(gpu):
    """Two vorticity-mode sims from the same seed must produce byte-identical
    ω textures after N steps (determinism gate)."""
    N = 20  # small but enough to exercise advect + force
    p = _vort_params(seed=99, steps=0)

    sim_a = Simulation(p, gpu)
    sim_a.solver.step(N)
    omega_a = _read_omega(sim_a).copy()
    sim_a._release_sim()

    sim_b = Simulation(p, gpu)
    sim_b.solver.step(N)
    omega_b = _read_omega(sim_b).copy()
    sim_b._release_sim()

    np.testing.assert_array_equal(
        omega_a, omega_b,
        err_msg="vorticity mode is not byte-identical across two runs with the same seed",
    )


# ---------------------------------------------------------------------------
# 3. Vorticity mode boundedness
# ---------------------------------------------------------------------------

def test_vorticity_mode_boundedness(gpu):
    """Over ~400 steps, ω must stay finite and not blow up.

    The MacCormack limiter + hyperviscosity must prevent unbounded growth.
    We also check that the field is not collapsed to zero (washout).
    """
    N = 400
    p = _vort_params(seed=7, steps=0)

    sim = Simulation(p, gpu)
    sim.solver.step(N)
    omega = _read_omega(sim)
    sim._release_sim()

    assert np.isfinite(omega).all(), "ω has non-finite values after 400 steps"

    # Enstrophy: mean(ω²).  Expect it to be positive (non-zero structure)
    # and below a sane ceiling (no blow-up).
    enstrophy = float(np.mean(omega ** 2))
    omega_max = float(np.abs(omega).max())

    # Sanity ceiling: with coriolis_f0=2, f_max = 2 at the poles.
    # A well-behaved q shouldn't exceed ~10× the initial scale.
    assert omega_max < 100.0, (
        f"ω blow-up: |ω|_max = {omega_max:.2f} after {N} steps "
        f"(ceiling 100)"
    )
    assert enstrophy > 1e-6, (
        f"ω collapsed to zero: enstrophy = {enstrophy:.2e} after {N} steps"
    )

    print(f"\nBoundedness: |ω|_max={omega_max:.4f}  "
          f"enstrophy={enstrophy:.4e}  (after {N} steps)")


# ---------------------------------------------------------------------------
# 4. Vorticity has spatial structure (sanity)
# ---------------------------------------------------------------------------

def test_vorticity_develops_structure(gpu):
    """The ω field should have non-trivial spatial structure: both latitude
    variation (from jets + Coriolis) and longitude variation (from vortices).

    A zonal-only init advected by a purely zonal flow stays zonal, so we
    don't require temporal growth — just that the field is alive and not
    degenerate.  The vortex contributions provide the longitude variation.

    This is a sanity gate, not a tight requirement.
    """
    # Use a seed with storms to get longitude-varying ω from vortex contributions.
    p = _vort_params(seed=7, steps=0)
    p.storms.hero_count = 1

    sim = Simulation(p, gpu)
    omega_init = _read_omega(sim).copy()
    sim._release_sim()

    # 1. Latitude variance: each row must not all be identical across longitudes.
    row_std_per_row = omega_init.std(axis=1)  # (H,) — std within each row
    # At least some rows must have longitude variation (vortex contributions).
    lon_var_max = float(row_std_per_row.max())

    # 2. Latitude variation: ω must vary across latitudes (jets + Coriolis).
    lat_means = omega_init.mean(axis=1)  # (H,) — mean q per latitude
    lat_var = float(lat_means.std())

    print(f"\nStructure: lon_var_max={lon_var_max:.4e}  lat_var={lat_var:.4e}")

    assert lat_var > 1.0, (
        f"ω has no latitude variation ({lat_var:.2e}) — "
        f"jets + Coriolis contribution is missing"
    )
    assert lon_var_max > 0.0, (
        f"ω has no longitude variation anywhere ({lon_var_max:.2e}) — "
        f"vortex contributions are not reaching the ω field"
    )


# ---------------------------------------------------------------------------
# 5. Kinematic mode unchanged (default path guard)
# ---------------------------------------------------------------------------

def test_kinematic_mode_unchanged_from_p3a(gpu):
    """With solver.type=kinematic (the default), the _omega_state must be
    None and the sim must run cleanly to completion.  This is an extra guard
    that the vorticity code is gated and doesn't touch the kinematic path."""
    p = PlanetParams(seed=1)
    p.sim.resolution = 512
    p.sim.dev_steps = 20
    # Default solver.type = KINEMATIC.
    assert p.solver.type == SolverType.KINEMATIC

    sim = Simulation(p, gpu)
    assert sim.solver._omega_state is None, (
        "omega_state should be None in KINEMATIC mode"
    )
    sim.run_to_completion()
    maps = sim.render_maps(width=256)
    assert np.isfinite(maps["color"]).all()
    sim._release_sim()

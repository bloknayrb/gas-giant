"""GPU tests for P6b: patch (AE) Laplacian sign-lock.

Tests
-----
1. patch_laplacian_glsl_sign_lock — GPU laplacianPsi compiled with DOMAIN=1 on
   Y_2^0 = 3cos²ρ−1 (ρ = colatitude = rho = length(patchStFromPix)) must give
   eigenvalue ≈ −6; 5% tolerance.  This is the unfakeable gate: wrong AE
   coefficients or wrong clamping cause a significant eigenvalue error.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_patch_laplacian_shader(gpu, N: int, rho_max: float):
    """Compile a compute shader that evaluates the patch laplacianPsi.

    Uses DOMAIN=1 so common.glsl declares u_rho_max and POLE_SIGN=1.0,
    and laplacian.glsl takes the #else (patch) branch.
    """
    import importlib.resources as ir  # noqa: PLC0415

    pkg = "gasgiant.sim.kernels"
    common_src    = ir.files(pkg).joinpath("common.glsl").read_text(encoding="utf-8")
    laplacian_src = ir.files(pkg).joinpath("laplacian.glsl").read_text(encoding="utf-8")

    full_src = (
        "#version 430\n"
        "layout(local_size_x = 16, local_size_y = 16) in;\n"
        "#define DOMAIN 1\n"
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
    prog = gpu.ctx.compute_shader(full_src)
    prog["u_rho_max"] = rho_max
    return prog


# ---------------------------------------------------------------------------
# 1. Patch Laplacian GPU sign-lock test
# ---------------------------------------------------------------------------

def test_patch_laplacian_glsl_sign_lock(gpu):
    """GPU patch laplacianPsi on Y_2^0 = 3cos²ρ−1 must give eigenvalue ≈ −6.

    ρ = colatitude = length(patchStFromPix).  Y_2^0 in colatitude coordinates:
      Y_2^0(ρ) = 3cos²ρ − 1
    (cos(ρ) = sin(lat) since ρ = π/2 − lat for the north patch).

    Eigenvalue −6 is confirmed by vorticity_ref.laplacian_patch (numpy
    reference, validated in test_vorticity_ref.py).  The 5% tolerance matches
    the existing equirect sign-lock test.  Interior mask: rho ∈ [3°, 30°]
    excludes the pole pixel and the patch edges where clamping dominates.
    """
    N = 128          # patch is N × N pixels; 128 is adequate for 5% accuracy
    rho_max = np.deg2rad(35.0)   # patch covers ±35° of colatitude

    # Build Y_2^0 = 3cos²ρ−1 field on the patch grid (CPU).
    # patchStFromPix: st = (pixPos/size * 2 - 1) * rho_max
    # pixel-centre pixPos = np.arange(N) + 0.5
    idx = (np.arange(N) + 0.5) / N * 2.0 - 1.0   # [-1+.., 1-..]
    coords = idx * rho_max   # s or t at pixel centres (radians)

    ss, tt = np.meshgrid(coords, coords)   # (N, N); ss varies along cols (x)
    rho2d = np.hypot(ss, tt)               # colatitude (N, N)

    Y = (3.0 * np.cos(rho2d) ** 2 - 1.0).astype(np.float32)   # (N, N)

    # Upload to GPU texture (R32F, single channel).
    field_tex = gpu.texture2d((N, N), 1, "f4", data=Y[:, :, np.newaxis])
    out_tex   = gpu.texture2d((N, N), 1, "f4")

    prog = _build_patch_laplacian_shader(gpu, N, rho_max)
    field_tex.use(location=0)
    prog["u_field"] = 0
    prog["u_size"]  = (N, N)
    out_tex.bind_to_image(0, read=False, write=True)

    gx = (N + 15) // 16
    gy = (N + 15) // 16
    prog.run(gx, gy, 1)
    gpu.ctx.memory_barrier()

    lap_gpu = gpu.read_texture(out_tex)[..., 0]   # (N, N)

    field_tex.release()
    out_tex.release()

    # Eigenvalue: lap / Y in the interior (exclude pole & edges where clamping
    # distorts the stencil).  rho ∈ [3°, 30°] keeps us well away from rho=0
    # (Y_2^0 → 2, not ideal for division) and from the patch boundary.
    rho_min_mask = np.deg2rad(3.0)
    rho_max_mask = np.deg2rad(30.0)
    interior = (rho2d > rho_min_mask) & (rho2d < rho_max_mask)

    Y_int   = Y[interior]
    lap_int = lap_gpu[interior]

    # Additional guard: exclude pixels where |Y| is too small (avoids noise amplification
    # near the ring where Y_2^0 = 0, i.e. rho ~ arccos(1/sqrt(3)) ≈ 54.7°, well outside
    # our mask, but keep |Y| > 0.05 for safety).
    nz = np.abs(Y_int) > 0.05
    ratio = lap_int[nz] / Y_int[nz]
    eigenvalue = float(np.median(ratio))
    error = abs(eigenvalue - (-6.0)) / 6.0

    print(
        f"\nGPU patch Y_2^0 eigenvalue: {eigenvalue:.4f}  expected: -6.000  "
        f"relative error: {error:.4f}"
    )

    assert error < 0.05, (
        f"GPU patch laplacianPsi eigenvalue error {error:.4%} > 5% — "
        f"check AE coefficients or clamping in laplacian.glsl #else branch "
        f"(measured eigenvalue: {eigenvalue:.4f})"
    )

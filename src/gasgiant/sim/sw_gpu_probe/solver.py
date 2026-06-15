"""GPU state container for the M0.5 2-layer shallow-water probe.

Field layout
------------
All textures use (width, height) = (W, H) for cell-centred fields and
(W, H+1) for meridional (v) face fields.

  h1, u1  : layer-1 height and zonal velocity  — (W, H)
  v1      : layer-1 meridional velocity         — (W, H+1)
  h2, u2  : layer-2 height and zonal velocity  — (W, H)
  v2      : layer-2 meridional velocity         — (W, H+1)
  h_eq1   : layer-1 equilibrium height          — (W, H)
  h_eq2   : layer-2 equilibrium height          — (W, H)

upload(name, arr) / download(name) work with (H, W) NumPy arrays for
cell-centred fields and (H+1, W) for v-face fields, matching the
(row, col) convention used by the CPU sw_spike solver.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import moderngl
    from gasgiant.gl.context import GpuContext

_KERNELS = "gasgiant.sim.kernels"
_GROUP = 16


def _set(prog: "moderngl.ComputeShader", name: str, value) -> None:
    """Set a uniform if the compiler kept it."""
    with contextlib.suppress(KeyError):
        prog[name].value = value


@dataclass
class SwpState:
    """Named R32F textures for the 2-layer shallow-water GPU probe."""

    gpu: GpuContext
    W: int
    H: int
    tex: dict[str, moderngl.Texture] = field(default_factory=dict)

    @classmethod
    def create(cls, gpu: GpuContext, W: int, H: int) -> SwpState:
        """Allocate all field textures.  No data is uploaded."""
        st = cls(gpu=gpu, W=W, H=H)

        # Cell-centred fields — texture size (W, H)
        for name in ("h1", "u1", "h2", "u2", "h_eq1", "h_eq2"):
            st.tex[name] = gpu.texture2d((W, H), components=1, dtype="f4")

        # Meridional face fields — texture size (W, H+1)
        for name in ("v1", "v2"):
            st.tex[name] = gpu.texture2d((W, H + 1), components=1, dtype="f4")

        return st

    def upload(self, name: str, arr: np.ndarray) -> None:
        """Write a (H, W) or (H+1, W) float32 array into the named texture."""
        self.tex[name].write(arr.astype(np.float32).tobytes())

    def download(self, name: str) -> np.ndarray:
        """Read the named texture and return a (H, W) or (H+1, W) float32 array."""
        return self.gpu.read_texture(self.tex[name])[..., 0]


def run_divergence(
    gpu: "GpuContext",
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """GPU flux-form mass divergence ∇·(hu) at cell centres.

    Ports divergence_hu() from sw_spike/operators.py.

    Parameters
    ----------
    gpu : GpuContext
    h   : (H, W) float32 — cell-centred layer depth
    u   : (H, W) float32 — zonal velocity at cell centres (east face)
    v   : (H+1, W) float32 — meridional velocity at v-faces

    Returns
    -------
    (H, W) float32 — divergence field, same contract as the CPU function.
    """
    import moderngl  # noqa: PLC0415 — local import to avoid hard dep at module level

    h = np.asarray(h, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = h.shape
    ctx = gpu.ctx

    # Allocate input textures.
    tex_h = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_u = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_div = gpu.texture2d((W, H), components=1, dtype="f4")

    # Upload inputs (write expects raw bytes in texture order: row-major, (H,W)).
    tex_h.write(h.tobytes())
    tex_u.write(u.tobytes())
    tex_v.write(v.tobytes())

    # Compile (or reuse from module-level cache) the divergence kernel.
    k = gpu.compute(_KERNELS, "swp_divergence.comp")

    # Set uniforms.
    _set(k, "u_size", (W, H))

    # Bind samplers.
    tex_h.use(location=0)
    _set(k, "u_h", 0)
    tex_u.use(location=1)
    _set(k, "u_u", 1)
    tex_v.use(location=2)
    _set(k, "u_v", 2)

    # Bind output image.
    tex_div.bind_to_image(0, read=False, write=True)

    # Dispatch.
    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    # Download result.
    result = gpu.read_texture(tex_div)[..., 0]

    # Release temporaries.
    for tex in (tex_h, tex_u, tex_v, tex_div):
        tex.release()

    return result


def run_grad_montgomery(
    gpu: "GpuContext",
    h1: np.ndarray,
    h2: np.ndarray,
    gp: tuple[float, float],
) -> dict[str, np.ndarray]:
    """GPU Montgomery potentials and their face gradients for the 2-layer SWP.

    Ports montgomery_2layer() + grad_faces() from sw_spike/operators.py.

    Parameters
    ----------
    gpu : GpuContext
    h1  : (H, W) float32 — layer-1 depth
    h2  : (H, W) float32 — layer-2 depth
    gp  : (g1, g2) reduced-gravity pair

    Returns
    -------
    dict with keys:
      M1, M2   : (H, W)   — Montgomery potentials at cell centres
      gx1, gx2 : (H, W)   — zonal gradient at u-faces (east face)
      gy1, gy2 : (H+1, W) — meridional gradient at v-faces
    """
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    g1, g2 = float(gp[0]), float(gp[1])

    H, W = h1.shape
    ctx = gpu.ctx

    # Input textures — cell-centred (W, H)
    tex_h1 = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_h2 = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_h1.write(h1.tobytes())
    tex_h2.write(h2.tobytes())

    # Output textures — cell-centred (W, H)
    tex_M1  = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_M2  = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_gx1 = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_gx2 = gpu.texture2d((W, H), components=1, dtype="f4")

    # Output textures — v-face (W, H+1)
    tex_gy1 = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_gy2 = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    # Compile (or reuse) the kernel.
    k = gpu.compute(_KERNELS, "swp_grad_montgomery.comp")

    # Uniforms — names must match GLSL declarations exactly.
    _set(k, "u_size", (W, H))
    _set(k, "u_g1", g1)
    _set(k, "u_g2", g2)

    # Bind samplers (location matches uniform sampler2D binding order).
    tex_h1.use(location=0)
    _set(k, "u_h1", 0)
    tex_h2.use(location=1)
    _set(k, "u_h2", 1)

    # Bind output images (binding indices match layout qualifiers in GLSL).
    tex_M1.bind_to_image(0,  read=False, write=True)
    tex_M2.bind_to_image(1,  read=False, write=True)
    tex_gx1.bind_to_image(2, read=False, write=True)
    tex_gx2.bind_to_image(3, read=False, write=True)
    tex_gy1.bind_to_image(4, read=False, write=True)
    tex_gy2.bind_to_image(5, read=False, write=True)

    # Dispatch over (W, H+1) to cover all v-face rows.
    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    # Download results.
    result = {
        "M1":  gpu.read_texture(tex_M1)[...,  0],
        "M2":  gpu.read_texture(tex_M2)[...,  0],
        "gx1": gpu.read_texture(tex_gx1)[..., 0],
        "gx2": gpu.read_texture(tex_gx2)[..., 0],
        "gy1": gpu.read_texture(tex_gy1)[..., 0],
        "gy2": gpu.read_texture(tex_gy2)[..., 0],
    }

    # Release temporaries.
    for tex in (tex_h1, tex_h2, tex_M1, tex_M2, tex_gx1, tex_gx2, tex_gy1, tex_gy2):
        tex.release()

    return result


def run_continuity(
    gpu: "GpuContext",
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    dt: float,
    h_floor: float,
) -> np.ndarray:
    """GPU FCT two-pass mass-conserving continuity step.

    Ports continuity_step() from sw_spike/operators.py.

    Parameters
    ----------
    gpu     : GpuContext
    h       : (H, W) float32 — cell-centred layer depth
    u       : (H, W) float32 — zonal velocity at cell centres (east face)
    v       : (H+1, W) float32 — meridional velocity at v-faces
    dt      : float — timestep
    h_floor : float — positivity floor

    Returns
    -------
    (H, W) float32 — updated layer depth h_new.
    """
    h = np.asarray(h, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = h.shape
    ctx = gpu.ctx

    # Input textures
    tex_h = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_u = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    tex_h.write(h.tobytes())
    tex_u.write(u.tobytes())
    tex_v.write(v.tobytes())

    # Scratch textures (pass A outputs, pass B inputs)
    tex_h_low = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_cap   = gpu.texture2d((W, H), components=1, dtype="f4")

    # Output texture
    tex_h_new = gpu.texture2d((W, H), components=1, dtype="f4")

    # Compile both kernel variants
    k_a = gpu.compute(_KERNELS, "swp_continuity.comp", defines={"PASS": "0"})
    k_b = gpu.compute(_KERNELS, "swp_continuity.comp", defines={"PASS": "1"})

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP

    # ---------- Pass A ----------
    _set(k_a, "u_size",    (W, H))
    _set(k_a, "u_dt",      float(dt))
    _set(k_a, "u_h_floor", float(h_floor))

    tex_h.use(location=0); _set(k_a, "u_h", 0)
    tex_u.use(location=1); _set(k_a, "u_u", 1)
    tex_v.use(location=2); _set(k_a, "u_v", 2)

    tex_h_low.bind_to_image(0, read=False, write=True)
    tex_cap.bind_to_image(1,   read=False, write=True)

    k_a.run(gx, gy, 1)
    ctx.memory_barrier()

    # ---------- Pass B ----------
    _set(k_b, "u_size",    (W, H))
    _set(k_b, "u_dt",      float(dt))
    _set(k_b, "u_h_floor", float(h_floor))

    tex_h.use(location=0);     _set(k_b, "u_h",     0)
    tex_u.use(location=1);     _set(k_b, "u_u",     1)
    tex_v.use(location=2);     _set(k_b, "u_v",     2)
    tex_h_low.use(location=3); _set(k_b, "u_h_low", 3)
    tex_cap.use(location=4);   _set(k_b, "u_cap",   4)

    tex_h_new.bind_to_image(0, read=False, write=True)

    k_b.run(gx, gy, 1)
    ctx.memory_barrier()

    # Download result
    result = gpu.read_texture(tex_h_new)[..., 0]

    # Release temporaries
    for tex in (tex_h, tex_u, tex_v, tex_h_low, tex_cap, tex_h_new):
        tex.release()

    return result


def run_momentum(
    gpu: "GpuContext",
    M: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    f0: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU vector-invariant momentum update for one layer.

    Ports _layer_momentum() from sw_spike/solver.py (single layer).

    Parameters
    ----------
    gpu : GpuContext
    M   : (H, W) float32 — Montgomery potential for this layer
    u   : (H, W) float32 — zonal velocity at u-faces
    v   : (H+1, W) float32 — meridional velocity at v-faces
    f0  : float — Coriolis scale factor (f = f0*sin(phi))
    dt  : float — timestep

    Returns
    -------
    u_new : (H, W) float32
    v_new : (H+1, W) float32
    """
    M = np.asarray(M, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = u.shape
    ctx = gpu.ctx

    # Step 1: compute vorticity into a scratch texture using the existing kernel.
    tex_u_in  = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v_in  = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_M_in  = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_zeta  = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_u_out = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v_out = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    tex_u_in.write(u.tobytes())
    tex_v_in.write(v.tobytes())
    tex_M_in.write(M.tobytes())

    # --- Vorticity pass ---
    k_vort = gpu.compute(_KERNELS, "swp_vorticity.comp")
    _set(k_vort, "u_size", (W, H))
    tex_u_in.use(location=0); _set(k_vort, "u_u", 0)
    tex_v_in.use(location=1); _set(k_vort, "u_v", 1)
    tex_zeta.bind_to_image(0, read=False, write=True)
    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP
    k_vort.run(gx, gy, 1)
    ctx.memory_barrier()

    # --- Momentum pass ---
    k_mom = gpu.compute(_KERNELS, "swp_momentum.comp")
    _set(k_mom, "u_size", (W, H))
    _set(k_mom, "u_f0",   float(f0))
    _set(k_mom, "u_dt",   float(dt))

    tex_zeta.use(location=0);  _set(k_mom, "u_zeta", 0)
    tex_u_in.use(location=1);  _set(k_mom, "u_u",    1)
    tex_v_in.use(location=2);  _set(k_mom, "u_v",    2)
    tex_M_in.use(location=3);  _set(k_mom, "u_M",    3)

    tex_u_out.bind_to_image(0, read=False, write=True)
    tex_v_out.bind_to_image(1, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP
    k_mom.run(gx, gy, 1)
    ctx.memory_barrier()

    u_new = gpu.read_texture(tex_u_out)[..., 0]
    v_new = gpu.read_texture(tex_v_out)[..., 0]

    for tex in (tex_u_in, tex_v_in, tex_M_in, tex_zeta, tex_u_out, tex_v_out):
        tex.release()

    return u_new, v_new


def run_forcing(
    gpu: "GpuContext",
    fields: dict[str, np.ndarray],
    v1: np.ndarray,
    v2: np.ndarray,
    params: dict,
    f0: float = 0.0,
) -> dict[str, np.ndarray]:
    """GPU forcing pass: relaxation → drag → hypervisc → polar sponge → floor.

    Ports _apply_forcing() from sw_spike/solver.py for a 2-layer shallow-water state.

    Parameters
    ----------
    gpu    : GpuContext
    fields : dict with keys h1, h2, u1, u2, h_eq1, h_eq2 — (H, W) float32
    v1     : (H+1, W) float32 — layer-1 meridional velocity
    v2     : (H+1, W) float32 — layer-2 meridional velocity
    params : dict with keys tau_rad, tau_drag, nu4, h_floor (all float)
    f0     : unused (kept for API symmetry with run_momentum)

    Returns
    -------
    dict with keys h1, h2, u1, u2, v1, v2 — post-forcing fields.
    """
    h1    = np.asarray(fields["h1"],    dtype=np.float32)
    h2    = np.asarray(fields["h2"],    dtype=np.float32)
    u1    = np.asarray(fields["u1"],    dtype=np.float32)
    u2    = np.asarray(fields["u2"],    dtype=np.float32)
    h_eq1 = np.asarray(fields["h_eq1"], dtype=np.float32)
    h_eq2 = np.asarray(fields["h_eq2"], dtype=np.float32)
    v1    = np.asarray(v1, dtype=np.float32)
    v2    = np.asarray(v2, dtype=np.float32)

    H, W = h1.shape
    ctx = gpu.ctx

    tau_rad  = float(params.get("tau_rad",  0.0))
    tau_drag = float(params.get("tau_drag", 0.0))
    nu4      = float(params.get("nu4",      0.0))
    h_floor  = float(params.get("h_floor",  0.05))

    # ── allocate input textures ────────────────────────────────────────────
    tex_h1    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_h2    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_u1    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_u2    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v1    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_v2    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_heq1  = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_heq2  = gpu.texture2d((W, H),     components=1, dtype="f4")

    tex_h1.write(h1.tobytes())
    tex_h2.write(h2.tobytes())
    tex_u1.write(u1.tobytes())
    tex_u2.write(u2.tobytes())
    tex_v1.write(v1.tobytes())
    tex_v2.write(v2.tobytes())
    tex_heq1.write(h_eq1.tobytes())
    tex_heq2.write(h_eq2.tobytes())

    # ── scratch textures (output of pass 0, input to pass 1) ──────────────
    sc_h1 = gpu.texture2d((W, H),     components=1, dtype="f4")
    sc_h2 = gpu.texture2d((W, H),     components=1, dtype="f4")
    sc_u1 = gpu.texture2d((W, H),     components=1, dtype="f4")
    sc_u2 = gpu.texture2d((W, H),     components=1, dtype="f4")
    sc_v1 = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    sc_v2 = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    # ── final output textures ──────────────────────────────────────────────
    out_h1 = gpu.texture2d((W, H),     components=1, dtype="f4")
    out_h2 = gpu.texture2d((W, H),     components=1, dtype="f4")
    out_u1 = gpu.texture2d((W, H),     components=1, dtype="f4")
    out_u2 = gpu.texture2d((W, H),     components=1, dtype="f4")
    out_v1 = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    out_v2 = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    # Dispatch size: (W, H+1) to cover both cell-centre (j<H) and v-face (j=H) rows
    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP

    def _set_common_uniforms(k):
        _set(k, "u_size",     (W, H))
        _set(k, "u_tau_rad",  tau_rad)
        _set(k, "u_tau_drag", tau_drag)
        _set(k, "u_nu4",      nu4)
        _set(k, "u_h_floor",  h_floor)

    def _bind_input_samplers(k):
        tex_h1.use(location=0);   _set(k, "u_h1",    0)
        tex_h2.use(location=1);   _set(k, "u_h2",    1)
        tex_u1.use(location=2);   _set(k, "u_u1",    2)
        tex_u2.use(location=3);   _set(k, "u_u2",    3)
        tex_v1.use(location=4);   _set(k, "u_v1",    4)
        tex_v2.use(location=5);   _set(k, "u_v2",    5)
        tex_heq1.use(location=6); _set(k, "u_h_eq1", 6)
        tex_heq2.use(location=7); _set(k, "u_h_eq2", 7)

    # ── Pass 0: relax h + drag u2/v2 ──────────────────────────────────────
    k0 = gpu.compute(_KERNELS, "swp_forcing.comp", defines={"PASS0": "1"})
    _set_common_uniforms(k0)
    _bind_input_samplers(k0)

    sc_h1.bind_to_image(0, read=False, write=True)
    sc_h2.bind_to_image(1, read=False, write=True)
    sc_u1.bind_to_image(2, read=False, write=True)
    sc_u2.bind_to_image(3, read=False, write=True)
    sc_v1.bind_to_image(4, read=False, write=True)
    sc_v2.bind_to_image(5, read=False, write=True)

    k0.run(gx, gy, 1)
    ctx.memory_barrier()

    # ── Pass 1: hypervisc + sponge + floor ────────────────────────────────
    k1 = gpu.compute(_KERNELS, "swp_forcing.comp", defines={})
    _set_common_uniforms(k1)
    # Pass 1 needs h_eq for sponge h relaxation
    tex_heq1.use(location=6); _set(k1, "u_h_eq1", 6)
    tex_heq2.use(location=7); _set(k1, "u_h_eq2", 7)
    # Scratch fields (post relax+drag) as samplers for pass 1
    sc_h1.use(location=8);  _set(k1, "u_s_h1", 8)
    sc_h2.use(location=9);  _set(k1, "u_s_h2", 9)
    sc_u1.use(location=10); _set(k1, "u_s_u1", 10)
    sc_u2.use(location=11); _set(k1, "u_s_u2", 11)
    sc_v1.use(location=12); _set(k1, "u_s_v1", 12)
    sc_v2.use(location=13); _set(k1, "u_s_v2", 13)

    out_h1.bind_to_image(0, read=False, write=True)
    out_h2.bind_to_image(1, read=False, write=True)
    out_u1.bind_to_image(2, read=False, write=True)
    out_u2.bind_to_image(3, read=False, write=True)
    out_v1.bind_to_image(4, read=False, write=True)
    out_v2.bind_to_image(5, read=False, write=True)

    k1.run(gx, gy, 1)
    ctx.memory_barrier()

    # ── download results ───────────────────────────────────────────────────
    result = {
        "h1": gpu.read_texture(out_h1)[..., 0],
        "h2": gpu.read_texture(out_h2)[..., 0],
        "u1": gpu.read_texture(out_u1)[..., 0],
        "u2": gpu.read_texture(out_u2)[..., 0],
        "v1": gpu.read_texture(out_v1)[..., 0],
        "v2": gpu.read_texture(out_v2)[..., 0],
    }

    # ── release temporaries ───────────────────────────────────────────────
    for tex in (
        tex_h1, tex_h2, tex_u1, tex_u2, tex_v1, tex_v2, tex_heq1, tex_heq2,
        sc_h1, sc_h2, sc_u1, sc_u2, sc_v1, sc_v2,
        out_h1, out_h2, out_u1, out_u2, out_v1, out_v2,
    ):
        tex.release()

    return result


def run_vorticity(
    gpu: "GpuContext",
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """GPU relative vorticity ζ at corners (H+1, W).

    Ports vorticity() from sw_spike/operators.py.

    Parameters
    ----------
    gpu : GpuContext
    u   : (H, W) float32 — zonal velocity at cell centres
    v   : (H+1, W) float32 — meridional velocity at v-faces

    Returns
    -------
    (H+1, W) float32 — relative vorticity at corners; poles forced to 0.
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = u.shape
    ctx = gpu.ctx

    # Input textures.
    tex_u = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_zeta = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    tex_u.write(u.tobytes())
    tex_v.write(v.tobytes())

    # Compile (or reuse) the vorticity kernel.
    k = gpu.compute(_KERNELS, "swp_vorticity.comp")

    # Uniforms — names must match GLSL declarations exactly.
    _set(k, "u_size", (W, H))

    # Bind samplers.
    tex_u.use(location=0)
    _set(k, "u_u", 0)
    tex_v.use(location=1)
    _set(k, "u_v", 1)

    # Bind output image.
    tex_zeta.bind_to_image(0, read=False, write=True)

    # Dispatch over (W, H+1) to cover all corner rows.
    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    # Download result.
    result = gpu.read_texture(tex_zeta)[..., 0]

    # Release temporaries.
    for tex in (tex_u, tex_v, tex_zeta):
        tex.release()

    return result

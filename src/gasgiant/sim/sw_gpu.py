"""GPU texture state for the M1 single-layer shallow-water solver.

Field layout
------------
All textures use moderngl's (width, height) = (W, H) convention.
NumPy arrays are (H, W) for cell-centred fields and (H+1, W) for v-face fields,
matching the (row, col) convention used by the CPU shallow_water_ref solver.

  h  : layer depth            — texture (W, H),   numpy (H, W)
  u  : zonal velocity         — texture (W, H),   numpy (H, W)
  v  : meridional velocity    — texture (W, H+1), numpy (H+1, W)

Tasks 4-9 add kernels and dispatch helpers; this module stays minimal.
"""

from __future__ import annotations

import contextlib
import math
import os
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
class SwGpuState:
    """Named R32F textures for the single-layer shallow-water GPU solver."""

    gpu: GpuContext
    W: int
    H: int
    a: float
    gp: float
    omega: float
    tex: dict[str, moderngl.Texture] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        gpu: GpuContext,
        W: int,
        H: int,
        a: float,
        gp: float,
        omega: float,
    ) -> SwGpuState:
        """Allocate all field textures.  No data is uploaded."""
        st = cls(gpu=gpu, W=W, H=H, a=float(a), gp=float(gp), omega=float(omega))

        # Cell-centred fields — texture size (W, H)
        for name in ("h", "u"):
            st.tex[name] = gpu.texture2d((W, H), components=1, dtype="f4")

        # Meridional face field — texture size (W, H+1)
        st.tex["v"] = gpu.texture2d((W, H + 1), components=1, dtype="f4")

        return st

    # -- upload helpers -------------------------------------------------------

    def upload_h(self, arr: np.ndarray) -> None:
        """Write a (H, W) float32 array into the h texture."""
        self.tex["h"].write(arr.astype(np.float32).tobytes())

    def upload_u(self, arr: np.ndarray) -> None:
        """Write a (H, W) float32 array into the u texture."""
        self.tex["u"].write(arr.astype(np.float32).tobytes())

    def upload_v(self, arr: np.ndarray) -> None:
        """Write a (H+1, W) float32 array into the v texture."""
        self.tex["v"].write(arr.astype(np.float32).tobytes())

    # -- download helpers -----------------------------------------------------

    def download_h(self) -> np.ndarray:
        """Read the h texture and return a (H, W) float32 array."""
        return self.gpu.read_texture(self.tex["h"])[..., 0]

    def download_u(self) -> np.ndarray:
        """Read the u texture and return a (H, W) float32 array."""
        return self.gpu.read_texture(self.tex["u"])[..., 0]

    def download_v(self) -> np.ndarray:
        """Read the v texture and return a (H+1, W) float32 array."""
        return self.gpu.read_texture(self.tex["v"])[..., 0]


def run_divergence(
    gpu: "GpuContext",
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: float,
) -> np.ndarray:
    """GPU flux-form mass divergence ∇·(hu) at cell centres, shape (H, W).

    Spherical metric: (1/(a cosφ))[ ∂(hu)/∂λ + ∂(hv cosφ)/∂φ ].
    Ports divergence_hu() from shallow_water_ref.py exactly, with radius `a`.

    Parameters
    ----------
    gpu : GpuContext
    h   : (H, W) float32 — cell-centred layer depth
    u   : (H, W) float32 — zonal velocity at cell centres (east face)
    v   : (H+1, W) float32 — meridional velocity at v-faces
    a   : float — planetary radius

    Returns
    -------
    (H, W) float32 — divergence field.
    """
    h = np.asarray(h, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = h.shape
    ctx = gpu.ctx

    # Allocate input textures.
    tex_h = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_u = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_div = gpu.texture2d((W, H),   components=1, dtype="f4")

    # Upload inputs.
    tex_h.write(h.tobytes())
    tex_u.write(u.tobytes())
    tex_v.write(v.tobytes())

    # Compile (or reuse) the divergence kernel.
    k = gpu.compute(_KERNELS, "sw_divergence.comp")

    # Set uniforms.
    _set(k, "u_size", (W, H))
    _set(k, "u_a",    float(a))
    _set(k, "u_dlam", 2.0 * math.pi / W)
    _set(k, "u_dphi", math.pi / H)

    # Bind samplers.
    tex_h.use(location=0); _set(k, "u_h", 0)
    tex_u.use(location=1); _set(k, "u_u", 1)
    tex_v.use(location=2); _set(k, "u_v", 2)

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


def run_vorticity(
    gpu: "GpuContext",
    u: np.ndarray,
    v: np.ndarray,
    a: float,
) -> np.ndarray:
    """GPU relative vorticity ζ = (1/(a cosφ))[∂v/∂λ − ∂(u cosφ)/∂φ] at corners, shape (H+1, W).

    Ports vorticity() from shallow_water_ref.py exactly, including the /a radius factor.

    Parameters
    ----------
    gpu : GpuContext
    u   : (H, W) float32 — zonal velocity at cell centres
    v   : (H+1, W) float32 — meridional velocity at v-faces
    a   : float — planetary radius

    Returns
    -------
    (H+1, W) float32 — relative vorticity at corners.
    """
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = u.shape
    ctx = gpu.ctx

    # Allocate input and output textures.
    tex_u    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_zeta = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    # Upload inputs.
    tex_u.write(u.tobytes())
    tex_v.write(v.tobytes())

    # Compile (or reuse) the vorticity kernel.
    k = gpu.compute(_KERNELS, "sw_vorticity.comp")

    # Set uniforms.
    _set(k, "u_size", (W, H))
    _set(k, "u_a",    float(a))
    _set(k, "u_dlam", 2.0 * math.pi / W)
    _set(k, "u_dphi", math.pi / H)

    # Bind samplers.
    tex_u.use(location=0); _set(k, "u_u", 0)
    tex_v.use(location=1); _set(k, "u_v", 1)

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


def run_continuity(
    gpu: "GpuContext",
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    dt: float,
    h_floor: float,
) -> np.ndarray:
    """GPU FCT two-pass continuity step; returns updated h, shape (H, W).

    Ports continuity_step() from shallow_water_ref.py exactly, with radius `a`
    in the flux-divergence denominator (1/(a cosφ)).

    Parameters
    ----------
    gpu     : GpuContext
    h       : (H, W) float32 — cell-centred layer depth
    u       : (H, W) float32 — zonal velocity at cell centres
    v       : (H+1, W) float32 — meridional velocity at v-faces
    a       : float — planetary radius
    dt      : float — time step
    h_floor : float — positivity floor

    Returns
    -------
    (H, W) float32 — updated h.
    """
    h = np.asarray(h, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = h.shape
    ctx = gpu.ctx

    # Allocate input textures.
    tex_h = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_u = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    # Scratch textures for pass 0 outputs.
    tex_h_low = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_cap   = gpu.texture2d((W, H), components=1, dtype="f4")

    # Output texture.
    tex_h_new = gpu.texture2d((W, H), components=1, dtype="f4")

    # Upload inputs.
    tex_h.write(h.tobytes())
    tex_u.write(u.tobytes())
    tex_v.write(v.tobytes())

    # -- Pass 0: compute h_low and cap ------------------------------------
    k0 = gpu.compute(_KERNELS, "sw_continuity.comp", defines={"PASS": "0"})

    _set(k0, "u_size",    (W, H))
    _set(k0, "u_dt",      float(dt))
    _set(k0, "u_h_floor", float(h_floor))
    _set(k0, "u_a",       float(a))

    tex_h.use(location=0); _set(k0, "u_h", 0)
    tex_u.use(location=1); _set(k0, "u_u", 1)
    tex_v.use(location=2); _set(k0, "u_v", 2)

    tex_h_low.bind_to_image(0, read=False, write=True)
    tex_cap.bind_to_image(1,   read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP
    k0.run(gx, gy, 1)
    ctx.memory_barrier()

    # -- Pass 1: FCT-limited fluxes -> h_new -----------------------------
    k1 = gpu.compute(_KERNELS, "sw_continuity.comp", defines={"PASS": "1"})

    _set(k1, "u_size",    (W, H))
    _set(k1, "u_dt",      float(dt))
    _set(k1, "u_h_floor", float(h_floor))
    _set(k1, "u_a",       float(a))

    tex_h.use(location=0);     _set(k1, "u_h",     0)
    tex_u.use(location=1);     _set(k1, "u_u",     1)
    tex_v.use(location=2);     _set(k1, "u_v",     2)
    tex_h_low.use(location=3); _set(k1, "u_h_low", 3)
    tex_cap.use(location=4);   _set(k1, "u_cap",   4)

    tex_h_new.bind_to_image(0, read=False, write=True)

    k1.run(gx, gy, 1)
    ctx.memory_barrier()

    # Download result.
    result = gpu.read_texture(tex_h_new)[..., 0]

    # Release temporaries.
    for tex in (tex_h, tex_u, tex_v, tex_h_low, tex_cap, tex_h_new):
        tex.release()

    return result


def run_grad(
    gpu: "GpuContext",
    h: np.ndarray,
    gp: float,
    a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU face pressure gradient ∇(g'·h) on C-grid faces.

    Computes M = gp·h internally, then evaluates grad_faces(M):
      gx[j,i] = (M[j,i+1] − M[j,i]) / (a·cosφ_c·dλ)  at east faces (H, W)
      gy[j]   = (M[j-1,i] − M[j,i]) / (a·dφ)          at v-faces (H+1, W)
    Pole rows of gy are zeroed.

    Parameters
    ----------
    gpu : GpuContext
    h   : (H, W) float32 — cell-centred layer depth
    gp  : float — reduced gravity g'
    a   : float — planetary radius

    Returns
    -------
    (gx, gy) — (H, W) and (H+1, W) float32 arrays.
    """
    h = np.asarray(h, dtype=np.float32)

    H, W = h.shape
    ctx = gpu.ctx

    # Allocate input and output textures.
    tex_h  = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_gx = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_gy = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    # Upload input.
    tex_h.write(h.tobytes())

    # Compile (or reuse) the gradient kernel.
    k = gpu.compute(_KERNELS, "sw_grad.comp")

    # Set uniforms.
    _set(k, "u_size", (W, H))
    _set(k, "u_a",    float(a))
    _set(k, "u_dlam", 2.0 * math.pi / W)
    _set(k, "u_dphi", math.pi / H)
    _set(k, "u_gp",   float(gp))

    # Bind sampler.
    tex_h.use(location=0); _set(k, "u_h", 0)

    # Bind output images.
    tex_gx.bind_to_image(0, read=False, write=True)
    tex_gy.bind_to_image(1, read=False, write=True)

    # Dispatch over (W, H+1) to cover all v-face rows.
    gx_groups = (W + _GROUP - 1) // _GROUP
    gy_groups = (H + 1 + _GROUP - 1) // _GROUP
    k.run(gx_groups, gy_groups, 1)
    ctx.memory_barrier()

    # Download results.
    result_gx = gpu.read_texture(tex_gx)[..., 0]
    result_gy = gpu.read_texture(tex_gy)[..., 0]

    # Release temporaries.
    for tex in (tex_h, tex_gx, tex_gy):
        tex.release()

    return result_gx, result_gy


def _upload_href(gpu: "GpuContext", H_ref_lat: np.ndarray, H: int):
    """Upload H_ref_lat (H,) as a (1,H) R32F texture sampled by row index.

    Cast to float32 before upload: reference_depth returns float64.
    """
    href = np.asarray(H_ref_lat, dtype=np.float32).reshape(H)
    if href.shape != (H,):
        raise ValueError(f"H_ref_lat must be shape ({H},), got {href.shape}")
    tex = gpu.texture2d((1, H), components=1, dtype="f4")
    tex.write(href.tobytes())
    return tex


def run_helmholtz_apply(
    gpu: "GpuContext",
    dh: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    a: float,
) -> np.ndarray:
    """GPU L_sym(dh) = dh - (theta*dt)^2*gp * div_H(grad(dh)), shape (H, W).

    Ports helmholtz_apply() from shallow_water_ref.py exactly (radius `a`).
    """
    dh = np.asarray(dh, dtype=np.float32)
    H, W = dh.shape
    ctx = gpu.ctx
    alpha = (theta * dt) ** 2 * gp

    tex_dh   = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_out  = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_dh.write(dh.tobytes())
    tex_href = _upload_href(gpu, H_ref_lat, H)

    k = gpu.compute(_KERNELS, "sw_helmholtz_apply.comp")
    _set(k, "u_size",  (W, H))
    _set(k, "u_alpha", float(alpha))
    _set(k, "u_a",     float(a))
    _set(k, "u_dlam",  2.0 * math.pi / W)
    _set(k, "u_dphi",  math.pi / H)
    tex_dh.use(location=0);   _set(k, "u_dh",   0)
    tex_href.use(location=1); _set(k, "u_Href", 1)
    tex_out.bind_to_image(0, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    result = gpu.read_texture(tex_out)[..., 0]
    for tex in (tex_dh, tex_out, tex_href):
        tex.release()
    return result


def run_helmholtz_residual(
    gpu: "GpuContext",
    dh: np.ndarray,
    rhs: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    a: float,
) -> np.ndarray:
    """GPU Helmholtz residual L_sym(dh) - rhs, shape (H, W)."""
    dh = np.asarray(dh, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    H, W = dh.shape
    ctx = gpu.ctx
    alpha = (theta * dt) ** 2 * gp

    tex_dh   = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_rhs  = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_out  = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_dh.write(dh.tobytes())
    tex_rhs.write(rhs.tobytes())
    tex_href = _upload_href(gpu, H_ref_lat, H)

    k = gpu.compute(_KERNELS, "sw_helmholtz_residual.comp")
    _set(k, "u_size",  (W, H))
    _set(k, "u_alpha", float(alpha))
    _set(k, "u_a",     float(a))
    _set(k, "u_dlam",  2.0 * math.pi / W)
    _set(k, "u_dphi",  math.pi / H)
    tex_dh.use(location=0);   _set(k, "u_dh",   0)
    tex_href.use(location=1); _set(k, "u_Href", 1)
    tex_rhs.use(location=2);  _set(k, "u_rhs",  2)
    tex_out.bind_to_image(0, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    result = gpu.read_texture(tex_out)[..., 0]
    for tex in (tex_dh, tex_rhs, tex_out, tex_href):
        tex.release()
    return result


def run_helmholtz_sor(
    gpu: "GpuContext",
    rhs: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    a: float,
    n_iters: int,
    sor_omega: float,
    dh0: np.ndarray | None = None,
) -> np.ndarray:
    """GPU fixed-count red/black SOR for L_sym dh = rhs, shape (H, W).

    Ports helmholtz_sor() from shallow_water_ref.py: exactly n_iters sweeps,
    each a red sweep (memory_barrier) then a black sweep (memory_barrier) with
    the analytic diagonal D.  Starts from zeros unless dh0 is supplied.
    The red and black kernels write IN PLACE into the same dh texture (each
    color reads only the other color + itself, so the update is race-free).
    """
    rhs = np.asarray(rhs, dtype=np.float32)
    H, W = rhs.shape
    ctx = gpu.ctx
    alpha = (theta * dt) ** 2 * gp

    dh_init = (np.zeros((H, W), dtype=np.float32)
               if dh0 is None else np.asarray(dh0, dtype=np.float32))

    tex_dh   = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_rhs  = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_dh.write(dh_init.tobytes())
    tex_rhs.write(rhs.tobytes())
    tex_href = _upload_href(gpu, H_ref_lat, H)

    k_red   = gpu.compute(_KERNELS, "sw_helmholtz_sor.comp", defines={"COLOR": "0"})
    k_black = gpu.compute(_KERNELS, "sw_helmholtz_sor.comp", defines={"COLOR": "1"})

    dlam, dphi = 2.0 * math.pi / W, math.pi / H
    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP

    def _bind(k):
        _set(k, "u_size",      (W, H))
        _set(k, "u_alpha",     float(alpha))
        _set(k, "u_a",         float(a))
        _set(k, "u_dlam",      dlam)
        _set(k, "u_dphi",      dphi)
        _set(k, "u_sor_omega", float(sor_omega))
        tex_dh.use(location=0);   _set(k, "u_dh",   0)
        tex_href.use(location=1); _set(k, "u_Href", 1)
        tex_rhs.use(location=2);  _set(k, "u_rhs",  2)
        tex_dh.bind_to_image(0, read=True, write=True)

    for _ in range(int(n_iters)):
        for k in (k_red, k_black):
            _bind(k)
            k.run(gx, gy, 1)
            ctx.memory_barrier()

    result = gpu.read_texture(tex_dh)[..., 0]
    for tex in (tex_dh, tex_rhs, tex_href):
        tex.release()
    return result


class SwGpuSolver:
    """Resident-texture single-layer shallow-water solver (M1).

    All fields stay on the GPU as R32F textures; ``step()`` dispatches
    pre-compiled kernels in CPU-reference order with ping-pong scratch textures
    and ``ctx.memory_barrier()`` between dependent dispatches.  No CPU round-trip
    occurs during ``step()``.

    Usage::

        sg = SwGpuSolver.from_williamson2(gpu, W=128, H=64, a=1.0, ...)
        sg.step()
        h, u, v = sg.download_state()
    """

    def __init__(
        self,
        gpu: "GpuContext",
        W: int,
        H: int,
        a: float,
        gp: float,
        omega: float,
        dt: float,
        h_floor: float = 0.05,
        *,
        semi_implicit: bool = False,
        theta: float = 0.5,
        sor_omega: float = 1.7,
        helmholtz_iters: int = 200,
        picard_iters: int = 3,
        dt_multiplier: float = 1.0,
        n_layers: int = 1,
        gp1: float | None = None,
        gp2: float = 0.0,
        tau_rad: float = 0.0,
        tau_drag: float = 0.0,
        nu4: float = 0.0,
        sponge_rate: float = 0.0,
    ) -> None:
        self.gpu = gpu
        self.ctx = gpu.ctx
        self.W = W
        self.H = H
        self.a = float(a)
        self.gp = float(gp)
        self.omega = float(omega)
        self.dt = float(dt)
        self.h_floor = float(h_floor)

        # -- M3 2-layer parameters ------------------------------------------
        # n_layers==1 is the unperturbed M1 path; n_layers==2 enables the
        # baroclinic 2-layer dispatch (_step_2layer).  The layer-2 textures and
        # the montgomery/forcing kernels are UNREACHABLE when n_layers==1.
        if n_layers not in (1, 2):
            raise ValueError(f"n_layers must be 1 or 2, got {n_layers!r}")
        self.n_layers = int(n_layers)
        # gp1 defaults to gp (single-layer reduced gravity) for back-compat.
        self.gp1 = float(gp if gp1 is None else gp1)
        self.gp2 = float(gp2)
        self.tau_rad = float(tau_rad)
        self.tau_drag = float(tau_drag)
        self.nu4 = float(nu4)
        self.sponge_rate = float(sponge_rate)
        # Optional per-step dispatch trace (dual-path gate asserts on this).
        self._record_dispatch = False
        self._dispatch_log: list[str] = []
        # Layer-2 state + h_eq, populated by from_2layer_state (n_layers==2).
        self._st2 = None

        # -- Semi-implicit path flag + inert parameter slots (T7 wires the SI path) --
        self.semi_implicit = bool(semi_implicit)
        self.theta = float(theta)
        self.sor_omega = float(sor_omega)
        self.helmholtz_iters = int(helmholtz_iters)
        self.picard_iters = int(picard_iters)
        self.dt_multiplier = float(dt_multiplier)
        # H_ref is populated by the constructor after upload (set to None here;
        # from_williamson2 sets it when semi_implicit=True).
        self.H_ref: np.ndarray | None = None

        # Dispatch group counts.
        self._gx_c = (W + _GROUP - 1) // _GROUP
        self._gy_c = (H + _GROUP - 1) // _GROUP
        self._gy_v = (H + 1 + _GROUP - 1) // _GROUP

        # Grid spacing.
        self._dlam = 2.0 * math.pi / W
        self._dphi = math.pi / H

        # -- Compile all kernels ONCE ----------------------------------------
        self._k_vort   = gpu.compute(_KERNELS, "sw_vorticity.comp")
        self._k_bern   = gpu.compute(_KERNELS, "sw_bernoulli.comp")
        self._k_grad   = gpu.compute(_KERNELS, "sw_grad.comp")
        self._k_mom    = gpu.compute(_KERNELS, "sw_momentum.comp")
        self._k_cont_a = gpu.compute(_KERNELS, "sw_continuity.comp", defines={"PASS": "0"})
        self._k_cont_b = gpu.compute(_KERNELS, "sw_continuity.comp", defines={"PASS": "1"})

        # -- Resident field textures ------------------------------------------
        self._tex_h = gpu.texture2d((W, H),     components=1, dtype="f4")
        self._tex_u = gpu.texture2d((W, H),     components=1, dtype="f4")
        self._tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4")

        # -- Scratch textures (pre-allocated, reused every step) --------------
        # Vorticity (H+1, W).
        self._tex_zeta  = gpu.texture2d((W, H + 1), components=1, dtype="f4")
        # Bernoulli (H, W).
        self._tex_B     = gpu.texture2d((W, H),     components=1, dtype="f4")
        # Bernoulli gradient.
        self._tex_gx    = gpu.texture2d((W, H),     components=1, dtype="f4")
        self._tex_gy    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
        # New u, v from momentum.
        self._tex_u_new = gpu.texture2d((W, H),     components=1, dtype="f4")
        self._tex_v_new = gpu.texture2d((W, H + 1), components=1, dtype="f4")
        # New h from continuity (pass A/B scratch).
        self._tex_h_low = gpu.texture2d((W, H),     components=1, dtype="f4")
        self._tex_cap   = gpu.texture2d((W, H),     components=1, dtype="f4")
        self._tex_h_new = gpu.texture2d((W, H),     components=1, dtype="f4")

    # -- Public constructors --------------------------------------------------

    @classmethod
    def from_williamson2(
        cls,
        gpu: "GpuContext",
        W: int,
        H: int,
        a: float,
        omega: float,
        u0: float,
        gp: float,
        h0: float,
        h_floor: float = 0.05,
        *,
        semi_implicit: bool = False,
        theta: float = 0.5,
        sor_omega: float = 1.7,
        helmholtz_iters: int = 200,
        picard_iters: int = 3,
        dt_multiplier: float = 1.0,
        n_layers: int = 1,
    ) -> "SwGpuSolver":
        """Build a SwGpuSolver from the analytic Williamson-2 initial condition.

        Produces the same initial h, u, v and dt as
        ``ref.williamson2_state(W, H, a, omega, u0, gp, h0)``.

        ``n_layers`` defaults to 1 (the M1 single-layer path); it is accepted so
        the dual-path gate can build the M1 solver explicitly.  n_layers==2 is
        constructed via :meth:`from_2layer_state`, not this analytic single-layer IC.
        """
        if n_layers != 1:
            raise ValueError(
                "from_williamson2 builds the single-layer IC; use from_2layer_state "
                "for n_layers==2"
            )
        from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
        st = ref.williamson2_state(W=W, H=H, a=a, omega=omega, u0=u0, gp=gp,
                                   h0=h0, h_floor=h_floor)
        sg = cls(
            gpu=gpu, W=W, H=H, a=a, gp=gp, omega=omega,
            dt=st.dt, h_floor=h_floor,
            semi_implicit=semi_implicit,
            theta=theta,
            sor_omega=sor_omega,
            helmholtz_iters=helmholtz_iters,
            picard_iters=picard_iters,
            dt_multiplier=dt_multiplier,
            n_layers=1,
        )
        sg._tex_h.write(st.h.astype(np.float32).tobytes())
        sg._tex_u.write(st.u.astype(np.float32).tobytes())
        sg._tex_v.write(st.v.astype(np.float32).tobytes())
        # Store initial velocity fields for velocity_l2_drift().
        sg.u_init = st.u.astype(np.float32).copy()
        sg.v_init = st.v.astype(np.float32).copy()
        # Compute and store H_ref (latitude profile) when semi-implicit path is requested.
        if semi_implicit:
            sg.H_ref = ref.reference_depth(st.h.astype(np.float32))
        return sg

    # -- Public I/O -----------------------------------------------------------

    def download_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (h, u, v) as numpy arrays: (H,W), (H,W), (H+1,W)."""
        h = self.gpu.read_texture(self._tex_h)[..., 0]
        u = self.gpu.read_texture(self._tex_u)[..., 0]
        v = self.gpu.read_texture(self._tex_v)[..., 0]
        return h, u, v

    # -- Diagnostic helpers ---------------------------------------------------

    def total_mass(self) -> float:
        """Global mass integral: Σ h · cosφ · a² dλ dφ."""
        from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
        h, _, _ = self.download_state()
        g = ref.Grid(self.W, self.H, self.a)
        return float(np.sum(h * g.cos_c[:, None]) * self.a * self.a * g.dlam * g.dphi)

    def velocity_l2_drift(self) -> float:
        """RMS drift from initial velocity: sqrt(mean((u-u_init)²) + mean((v-v_init)²)).

        Mirrors shallow_water_ref.velocity_l2_drift(): u and v live on different
        staggered grids (H,W) and (H+1,W), averaged separately and combined.
        Requires u_init/v_init to have been set by from_williamson2().
        """
        _, u, v = self.download_state()
        du = u - self.u_init
        dv = v - self.v_init
        return float(np.sqrt(np.mean(du * du) + np.mean(dv * dv)))

    def total_energy(self) -> float:
        """Global total energy: Σ [½h(u² + v_c²) + ½g'h²] cosφ · a² dλ dφ.

        Mirrors shallow_water_ref.total_energy() with f64 reduction of f32 readback.
        v_c is the cell-centred meridional velocity averaged from the two bounding v-faces.

        NOTE: This is mass-CLOSED + energy-MONITORED (bounded drift gate).
        True §6 budget closure (forcing == dissipation) is an M3 gate; M1 has no forcing.
        The only energy sink here is the FCT floor + numerical truncation.
        """
        from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
        h, u, v = self.download_state()
        # Promote to f64 for accurate global reduction.
        h = h.astype(np.float64)
        u = u.astype(np.float64)
        v = v.astype(np.float64)
        H = self.H
        v_c = 0.5 * (v[0:H] + v[1:H + 1])          # cell-centred v, shape (H, W)
        ke = 0.5 * h * (u * u + v_c * v_c)
        pe = 0.5 * self.gp * h * h
        g = ref.Grid(self.W, H, self.a)
        return float(np.sum((ke + pe) * g.cos_c[:, None]) * self.a * self.a * g.dlam * g.dphi)

    def total_potential_enstrophy(self) -> float:
        """Global potential enstrophy diagnostic: Σ ½(ζ+f)²/h_corner · cosφ_corner · a² dλ dφ.

        Diagnostic only — no hard closure gate in M1 (no forcing/drag).
        h_corner is averaged from adjacent cells and floored to avoid division by zero.
        Mirrors shallow_water_ref.total_potential_enstrophy() with f64 reduction.
        """
        from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
        h, u, v = self.download_state()
        h = h.astype(np.float64)
        u = u.astype(np.float64)
        v = v.astype(np.float64)
        H, W = self.H, self.W
        g = ref.Grid(W, H, self.a)

        # Relative vorticity at corners (H+1, W) via CPU reference (f64 inputs).
        zeta = ref.vorticity(u, v, g)

        # Planetary vorticity at v-faces/corners.
        f_v = 2.0 * self.omega * np.sin(g.phi_v)[:, None] * np.ones((1, W))
        abs_vort = zeta + f_v

        # h at corners: meridional average of adjacent cells (pole rows floored).
        h_corner = np.full((H + 1, W), self.h_floor)
        h_corner[1:H] = 0.5 * (h[0:H - 1] + h[1:H])
        h_corner = np.maximum(h_corner, self.h_floor)

        cos_v = g.cos_v[:, None] * np.ones((1, W))
        ens = 0.5 * abs_vort * abs_vort / h_corner * cos_v
        return float(np.sum(ens) * self.a * self.a * g.dlam * g.dphi)

    # -- Checkpoint I/O -------------------------------------------------------

    def save_checkpoint(self, path: str | os.PathLike) -> None:
        """Save solver state to a .npz file (version 2).

        Version 2 stores all M1 scalar parameters (W, H, a, gp, omega, dt,
        h_floor) PLUS the M2 semi-implicit parameters (semi_implicit, theta,
        sor_omega, helmholtz_iters, picard_iters, dt_multiplier) and H_ref
        (the latitude reference-depth profile, or absent when None).

        Files are always written as version=2; version=1 files written by
        older code can still be loaded via :meth:`load_checkpoint`.

        Parameters
        ----------
        path : path-like
            Destination file (a ``.npz`` extension is appended by ``np.savez``
            if not already present).
        """
        # Layer 1 is always the resident (_tex_h/u/v) state.  For n_layers==1
        # this is the only layer; the file stays VERSION 2 (back-compatible).
        # For n_layers==2 we bump to VERSION 3 and add layer-indexed keys
        # (h1/u1/v1 == the version-2 h/u/v aliases, plus h2/u2/v2, gp1/gp2, the
        # forcing params, and h_eq1/h_eq2 when present).
        h, u, v = self.download_state()
        version = 3 if self.n_layers == 2 else 2
        arrays: dict = dict(
            version=np.int32(version),
            W=np.int32(self.W),
            H=np.int32(self.H),
            a=np.float64(self.a),
            gp=np.float64(self.gp),
            omega=np.float64(self.omega),
            dt=np.float64(self.dt),
            h_floor=np.float64(self.h_floor),
            h=h.astype(np.float32),
            u=u.astype(np.float32),
            v=v.astype(np.float32),
            # M2 SI parameters
            semi_implicit=np.bool_(self.semi_implicit),
            theta=np.float64(self.theta),
            sor_omega=np.float64(self.sor_omega),
            helmholtz_iters=np.int32(self.helmholtz_iters),
            picard_iters=np.int32(self.picard_iters),
            dt_multiplier=np.float64(self.dt_multiplier),
            n_layers=np.int32(self.n_layers),
        )
        # H_ref: persist only if present (semi_implicit=True sets it).
        if self.H_ref is not None:
            arrays["H_ref"] = np.asarray(self.H_ref, dtype=np.float32)
        # M3 2-layer keys.
        if self.n_layers == 2:
            h1, u1, v1, h2, u2, v2 = self.download_state_2layer()
            arrays.update(
                gp1=np.float64(self.gp1), gp2=np.float64(self.gp2),
                tau_rad=np.float64(self.tau_rad), tau_drag=np.float64(self.tau_drag),
                nu4=np.float64(self.nu4), sponge_rate=np.float64(self.sponge_rate),
                h1=h1.astype(np.float32), u1=u1.astype(np.float32), v1=v1.astype(np.float32),
                h2=h2.astype(np.float32), u2=u2.astype(np.float32), v2=v2.astype(np.float32),
            )
            if self._h_eq1 is not None:
                arrays["h_eq1"] = np.asarray(self._h_eq1, dtype=np.float32)
                arrays["h_eq2"] = np.asarray(self._h_eq2, dtype=np.float32)
        np.savez(path, **arrays)

    @classmethod
    def load_checkpoint(cls, gpu: "GpuContext", path: str | os.PathLike) -> "SwGpuSolver":
        """Reconstruct a :class:`SwGpuSolver` from a checkpoint.

        Accepts both **version 1** (M1 explicit solver — semi_implicit defaults
        to False) and **version 2** (M2 semi-implicit — restores SI params and
        H_ref so that continuation is bit-exact).

        Parameters
        ----------
        gpu  : GpuContext
        path : path-like
            Path to the ``.npz`` checkpoint file.

        Returns
        -------
        SwGpuSolver
            Solver with GPU field textures and all parameters pre-loaded.

        Raises
        ------
        ValueError
            If the checkpoint version is not 1 or 2.
        """
        # Context manager guarantees the .npz zip handle is released before the
        # caller (e.g. pytest tmp_path) tries to clean up — matters on Windows,
        # where a lingering handle blocks file removal (WinError 32).
        with np.load(path) as data:
            version = int(data["version"])
            if version not in (1, 2, 3):
                raise ValueError(
                    f"Unsupported checkpoint version {version!r}; expected 1, 2 or 3"
                )
            W = int(data["W"])
            H = int(data["H"])
            a = float(data["a"])
            gp = float(data["gp"])
            omega = float(data["omega"])
            dt = float(data["dt"])
            h_floor = float(data["h_floor"])
            # .astype returns fresh arrays that outlive the closed zip handle.
            h = data["h"].astype(np.float32)
            u = data["u"].astype(np.float32)
            v = data["v"].astype(np.float32)

            # M2 SI parameters — present in version 2 and 3.
            if version in (2, 3):
                semi_implicit = bool(data["semi_implicit"])
                theta = float(data["theta"])
                sor_omega = float(data["sor_omega"])
                helmholtz_iters = int(data["helmholtz_iters"])
                picard_iters = int(data["picard_iters"])
                dt_multiplier = float(data["dt_multiplier"])
                H_ref = (
                    data["H_ref"].astype(np.float32)
                    if "H_ref" in data
                    else None
                )
                if version == 3:
                    # M3 2-layer checkpoint: reconstruct via the CPU Sw2State +
                    # from_2layer_state so all layer-2 textures + h_eq are restored.
                    from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
                    g = ref.Grid(W, H, a)
                    heq1 = data["h_eq1"].astype(np.float32) if "h_eq1" in data else None
                    heq2 = data["h_eq2"].astype(np.float32) if "h_eq2" in data else None
                    st2 = ref.Sw2State(
                        g=g, omega=omega,
                        gp1=float(data["gp1"]), gp2=float(data["gp2"]),
                        h1=data["h1"].astype(np.float32),
                        u1=data["u1"].astype(np.float32),
                        v1=data["v1"].astype(np.float32),
                        h2=data["h2"].astype(np.float32),
                        u2=data["u2"].astype(np.float32),
                        v2=data["v2"].astype(np.float32),
                        dt=dt, h_floor=h_floor,
                        tau_rad=float(data["tau_rad"]),
                        tau_drag=float(data["tau_drag"]),
                        nu4=float(data["nu4"]),
                        sponge_rate=float(data["sponge_rate"]),
                        h_eq1=heq1, h_eq2=heq2,
                    )
                    return cls.from_2layer_state(gpu, st2)
            else:
                # version 1: explicit solver, no SI state.
                semi_implicit = False
                theta = 0.5
                sor_omega = 1.7
                helmholtz_iters = 200
                picard_iters = 3
                dt_multiplier = 1.0
                H_ref = None

        sg = cls(
            gpu=gpu,
            W=W,
            H=H,
            a=a,
            gp=gp,
            omega=omega,
            dt=dt,
            h_floor=h_floor,
            semi_implicit=semi_implicit,
            theta=theta,
            sor_omega=sor_omega,
            helmholtz_iters=helmholtz_iters,
            picard_iters=picard_iters,
            dt_multiplier=dt_multiplier,
        )

        # Upload saved fields to GPU textures (bit-exact float32).
        sg._tex_h.write(h.tobytes())
        sg._tex_u.write(u.tobytes())
        sg._tex_v.write(v.tobytes())

        # Restore initial velocity fields so velocity_l2_drift() stays callable.
        sg.u_init = u.copy()
        sg.v_init = v.copy()

        # Restore H_ref for SI continuation.
        sg.H_ref = H_ref

        return sg

    # -- Step -----------------------------------------------------------------

    def _si_grid(self):
        """Cached CPU-reference Grid for the SI anomaly assembly (built once)."""
        g = getattr(self, "_si_grid_cache", None)
        if g is None:
            from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
            g = ref.Grid(self.W, self.H, self.a)
            self._si_grid_cache = g
        return g

    def _step_semi_implicit(self) -> None:
        """One M2 semi-implicit step on the GPU (mirrors ref.step_semi_implicit).

        Predictor -> Picard loop (warm-started dh) over {helmholtz_rhs +
        helmholtz_sor} -> velocity_backsub -> conservative continuity -> assemble
        h_new = h + dh + (h_fct - h_linref), with a loud positivity guard.

        Uses self.dt * self.dt_multiplier as the SI dt.  All field math runs on
        the GPU kernels; the small per-step assembly (anomaly, max-floor) is a
        numpy op on f32 readbacks (matching the CPU's f64->the same algebra; the
        per-field tolerance test documents the f32 vs f64 gap).
        """
        gpu = self.gpu
        W, H, a, gp, omega = self.W, self.H, self.a, self.gp, self.omega
        theta = self.theta
        dt = self.dt * self.dt_multiplier
        h_floor = self.h_floor
        if self.H_ref is None:
            raise ValueError("_step_semi_implicit requires H_ref (semi_implicit=True)")
        H_ref = np.asarray(self.H_ref, dtype=np.float32)

        # Resident state.
        h = self.gpu.read_texture(self._tex_h)[..., 0]
        u = self.gpu.read_texture(self._tex_u)[..., 0]
        v = self.gpu.read_texture(self._tex_v)[..., 0]

        # 1. Predictor (no Coriolis).
        u_star, v_star = run_si_predictor(gpu, h, u, v, a, gp, dt, theta)

        # 3. Picard loop with deferred Coriolis; warm-start dh across iters.
        dh = np.zeros((H, W), dtype=np.float32)
        for _ in range(self.picard_iters):
            rhs = run_helmholtz_rhs(gpu, h, u, v, u_star, v_star, dh,
                                    H_ref, gp, omega, theta, dt, a)
            dh = run_helmholtz_sor(gpu, rhs, H_ref, gp, theta, dt, a,
                                   self.helmholtz_iters, self.sor_omega, dh0=dh)

        # 4. Back-substitution: implicit pressure of full height h + dh.
        h_impl = (h + dh).astype(np.float32)
        u_new, v_new = run_velocity_backsub(gpu, u_star, v_star, h_impl,
                                            gp, theta, dt, omega, a)

        # 5. Final height: matched theta-centered increment + explicit anomaly.
        h_fct = run_continuity_conservative(gpu, h, u_new, v_new, a, dt, h_floor)
        from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
        h_linref = h - dt * ref.divergence_helmholtz(
            u_new, v_new, H_ref.astype(np.float64), self._si_grid())
        anomaly = h_fct - h_linref
        h_raw = h + dh + anomaly
        ref.assert_positivity(h_raw, h_floor)        # shared CPU/GPU guard
        h_new = np.maximum(h_raw, h_floor).astype(np.float32)

        self._tex_h.write(h_new.tobytes())
        self._tex_u.write(u_new.astype(np.float32).tobytes())
        self._tex_v.write(v_new.astype(np.float32).tobytes())

    def step(self) -> None:
        """One full step in CPU-reference order (no CPU round-trip).

        For the semi-implicit path, delegate to _step_semi_implicit.

        Sequence (mirrors ref.step / ref.momentum_step order):
          1. vorticity(old u, v) → ζ
          2. bernoulli(old h, u, v) → B
          3. grad(B, gp=1.0) → gx, gy
          4. momentum-assembly(ζ, gx, gy, old u, v) → u_new, v_new
          5. continuity pass-A (old h, u_new, v_new) → h_low, cap
          6. continuity pass-B → h_new
          7. ping-pong: swap resident h/u/v ↔ h_new/u_new/v_new
        """
        if self.semi_implicit:
            self._step_semi_implicit()
            return

        if self.n_layers == 2:
            self._step_2layer()
            return

        ctx = self.ctx
        W, H = self.W, self.H
        a = self.a
        gp = self.gp
        omega = self.omega
        dt = self.dt
        h_floor = self.h_floor
        dlam = self._dlam
        dphi = self._dphi
        gx_c, gy_c, gy_v = self._gx_c, self._gy_c, self._gy_v

        # --- 1. Vorticity (old u, v) → ζ at corners (H+1, W) ---------------
        k = self._k_vort
        _set(k, "u_size", (W, H))
        _set(k, "u_a",    a)
        _set(k, "u_dlam", dlam)
        _set(k, "u_dphi", dphi)
        self._tex_u.use(location=0); _set(k, "u_u", 0)
        self._tex_v.use(location=1); _set(k, "u_v", 1)
        self._tex_zeta.bind_to_image(0, read=False, write=True)
        self._log("sw_vorticity.comp")
        k.run(gx_c, gy_v, 1)
        ctx.memory_barrier()

        # --- 2. Bernoulli potential (old h, u, v) → B at centres (H, W) ----
        k = self._k_bern
        _set(k, "u_size", (W, H))
        _set(k, "u_gp",   gp)
        self._tex_h.use(location=0); _set(k, "u_h", 0)
        self._tex_u.use(location=1); _set(k, "u_u", 1)
        self._tex_v.use(location=2); _set(k, "u_v", 2)
        self._tex_B.bind_to_image(0, read=False, write=True)
        self._log("sw_bernoulli.comp")
        k.run(gx_c, gy_c, 1)
        ctx.memory_barrier()

        # --- 3. grad(B, gp=1.0, a) → gx, gy (face gradients of B) ----------
        k = self._k_grad
        _set(k, "u_size", (W, H))
        _set(k, "u_a",    a)
        _set(k, "u_dlam", dlam)
        _set(k, "u_dphi", dphi)
        _set(k, "u_gp",   1.0)   # B already contains gp*h + ke; scale by 1
        self._tex_B.use(location=0); _set(k, "u_h", 0)
        self._tex_gx.bind_to_image(0, read=False, write=True)
        self._tex_gy.bind_to_image(1, read=False, write=True)
        self._log("sw_grad.comp")
        k.run(gx_c, gy_v, 1)
        ctx.memory_barrier()

        # --- 4. Momentum assembly → u_new, v_new ----------------------------
        k = self._k_mom
        _set(k, "u_size",  (W, H))
        _set(k, "u_omega", omega)
        _set(k, "u_dt",    dt)
        self._tex_zeta.use(location=0); _set(k, "u_zeta", 0)
        self._tex_gx.use(location=1);   _set(k, "u_gx",   1)
        self._tex_gy.use(location=2);   _set(k, "u_gy",   2)
        self._tex_u.use(location=3);    _set(k, "u_u",    3)
        self._tex_v.use(location=4);    _set(k, "u_v",    4)
        self._tex_u_new.bind_to_image(0, read=False, write=True)
        self._tex_v_new.bind_to_image(1, read=False, write=True)
        self._log("sw_momentum.comp")
        k.run(gx_c, gy_v, 1)
        ctx.memory_barrier()

        # --- 5. Continuity pass A (old h, new u/v) → h_low, cap ------------
        k = self._k_cont_a
        _set(k, "u_size",    (W, H))
        _set(k, "u_dt",      dt)
        _set(k, "u_h_floor", h_floor)
        _set(k, "u_a",       a)
        self._tex_h.use(location=0);     _set(k, "u_h", 0)
        self._tex_u_new.use(location=1); _set(k, "u_u", 1)
        self._tex_v_new.use(location=2); _set(k, "u_v", 2)
        self._tex_h_low.bind_to_image(0, read=False, write=True)
        self._tex_cap.bind_to_image(1,   read=False, write=True)
        self._log("sw_continuity.comp:0")
        k.run(gx_c, gy_c, 1)
        ctx.memory_barrier()

        # --- 6. Continuity pass B → h_new ------------------------------------
        k = self._k_cont_b
        _set(k, "u_size",    (W, H))
        _set(k, "u_dt",      dt)
        _set(k, "u_h_floor", h_floor)
        _set(k, "u_a",       a)
        self._tex_h.use(location=0);     _set(k, "u_h",     0)
        self._tex_u_new.use(location=1); _set(k, "u_u",     1)
        self._tex_v_new.use(location=2); _set(k, "u_v",     2)
        self._tex_h_low.use(location=3); _set(k, "u_h_low", 3)
        self._tex_cap.use(location=4);   _set(k, "u_cap",   4)
        self._tex_h_new.bind_to_image(0, read=False, write=True)
        self._log("sw_continuity.comp:1")
        k.run(gx_c, gy_c, 1)
        ctx.memory_barrier()

        # --- 7. Ping-pong: swap resident ↔ new ------------------------------
        self._tex_h, self._tex_h_new = self._tex_h_new, self._tex_h
        self._tex_u, self._tex_u_new = self._tex_u_new, self._tex_u
        self._tex_v, self._tex_v_new = self._tex_v_new, self._tex_v

    # -- Logging + M3 2-layer dispatch ---------------------------------------

    def _log(self, name: str) -> None:
        """Record a kernel dispatch when tracing is enabled (dual-path gate)."""
        if self._record_dispatch:
            self._dispatch_log.append(name)

    @classmethod
    def from_2layer_state(cls, gpu: "GpuContext", st) -> "SwGpuSolver":
        """Build a 2-layer (n_layers=2) GPU solver from a CPU Sw2State.

        Copies both layers' fields + h_eq + forcing params to the GPU.  The
        resident layer-1 textures (_tex_h/_tex_u/_tex_v) hold layer 1; layer 2
        and the h_eq targets live in a small dict (self._st2).
        """
        g = st.g
        sg = cls(
            gpu=gpu, W=g.W, H=g.H, a=g.a, gp=st.gp1, omega=st.omega,
            dt=st.dt, h_floor=st.h_floor,
            n_layers=2, gp1=st.gp1, gp2=st.gp2,
            tau_rad=st.tau_rad, tau_drag=st.tau_drag,
            nu4=st.nu4, sponge_rate=st.sponge_rate,
        )
        # Layer 1 -> resident textures.
        sg._tex_h.write(st.h1.astype(np.float32).tobytes())
        sg._tex_u.write(st.u1.astype(np.float32).tobytes())
        sg._tex_v.write(st.v1.astype(np.float32).tobytes())
        # Layer 2 + h_eq -> resident layer-2 textures.
        W, H = g.W, g.H
        st2 = {
            "h2": gpu.texture2d((W, H),     components=1, dtype="f4"),
            "u2": gpu.texture2d((W, H),     components=1, dtype="f4"),
            "v2": gpu.texture2d((W, H + 1), components=1, dtype="f4"),
        }
        st2["h2"].write(st.h2.astype(np.float32).tobytes())
        st2["u2"].write(st.u2.astype(np.float32).tobytes())
        st2["v2"].write(st.v2.astype(np.float32).tobytes())
        sg._st2 = st2
        sg._h_eq1 = None if st.h_eq1 is None else np.asarray(st.h_eq1, np.float32)
        sg._h_eq2 = None if st.h_eq2 is None else np.asarray(st.h_eq2, np.float32)
        return sg

    def download_state_2layer(self):
        """Return (h1, u1, v1, h2, u2, v2) for the 2-layer solver."""
        if self.n_layers != 2:
            raise ValueError("download_state_2layer requires n_layers==2")
        h1 = self.gpu.read_texture(self._tex_h)[..., 0]
        u1 = self.gpu.read_texture(self._tex_u)[..., 0]
        v1 = self.gpu.read_texture(self._tex_v)[..., 0]
        h2 = self.gpu.read_texture(self._st2["h2"])[..., 0]
        u2 = self.gpu.read_texture(self._st2["u2"])[..., 0]
        v2 = self.gpu.read_texture(self._st2["v2"])[..., 0]
        return h1, u1, v1, h2, u2, v2

    def _step_2layer(self) -> None:
        """One explicit 2-layer step (n_layers==2); mirrors ref.step_2layer.

        Order: montgomery(h1,h2) -> per-layer momentum_M -> per-layer
        conservative continuity -> positivity guard -> forcing.  Uses the
        a-aware production GLSL via the run_* orchestrators (diffed per-field vs
        the CPU ref at atol=2e-5).
        """
        from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415

        gpu, a = self.gpu, self.a
        h1 = self.gpu.read_texture(self._tex_h)[..., 0]
        u1 = self.gpu.read_texture(self._tex_u)[..., 0]
        v1 = self.gpu.read_texture(self._tex_v)[..., 0]
        h2 = self.gpu.read_texture(self._st2["h2"])[..., 0]
        u2 = self.gpu.read_texture(self._st2["u2"])[..., 0]
        v2 = self.gpu.read_texture(self._st2["v2"])[..., 0]

        # 1. Montgomery potentials from OLD h.
        self._log("sw_montgomery.comp")
        M1, M2 = run_montgomery(gpu, h1, h2, self.gp1, self.gp2)

        # 2-3. Per-layer momentum (vorticity -> bernoulli-M -> grad -> momentum).
        self._log("sw_momentum_M.layer1")
        u1n, v1n = run_momentum_M(gpu, M1, u1, v1, a, self.omega, self.dt)
        self._log("sw_momentum_M.layer2")
        u2n, v2n = run_momentum_M(gpu, M2, u2, v2, a, self.omega, self.dt)

        # 4-5. Per-layer conservative continuity (OLD h, NEW u/v).
        self._log("sw_continuity_conservative.layer1")
        h1n = run_continuity_conservative(gpu, h1, u1n, v1n, a, self.dt, self.h_floor)
        self._log("sw_continuity_conservative.layer2")
        h2n = run_continuity_conservative(gpu, h2, u2n, v2n, a, self.dt, self.h_floor)

        ref.assert_positivity(h1n, self.h_floor)
        ref.assert_positivity(h2n, self.h_floor)

        # 6. Forcing (sw_forcing.comp two-pass) — no-op when all params are off.
        g = ref.Grid(self.W, self.H, self.a)
        st = ref.Sw2State(
            g=g, omega=self.omega, gp1=self.gp1, gp2=self.gp2,
            h1=h1n, u1=u1n, v1=v1n, h2=h2n, u2=u2n, v2=v2n,
            dt=self.dt, h_floor=self.h_floor,
            tau_rad=self.tau_rad, tau_drag=self.tau_drag,
            nu4=self.nu4, sponge_rate=self.sponge_rate,
            h_eq1=self._h_eq1, h_eq2=self._h_eq2,
        )
        forced = run_forcing_2layer(gpu, st)
        self._log("sw_forcing.comp")

        # Write results back into resident textures.
        self._tex_h.write(np.ascontiguousarray(forced["h1"], np.float32).tobytes())
        self._tex_u.write(np.ascontiguousarray(forced["u1"], np.float32).tobytes())
        self._tex_v.write(np.ascontiguousarray(forced["v1"], np.float32).tobytes())
        self._st2["h2"].write(np.ascontiguousarray(forced["h2"], np.float32).tobytes())
        self._st2["u2"].write(np.ascontiguousarray(forced["u2"], np.float32).tobytes())
        self._st2["v2"].write(np.ascontiguousarray(forced["v2"], np.float32).tobytes())


def run_continuity_conservative(
    gpu: "GpuContext",
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    dt: float,
    h_floor: float,
) -> np.ndarray:
    """GPU conservative FCT continuity step; ports continuity_step_conservative().

    Five passes (s_pos -> cap -> Fx/Fy_tot -> s_tot -> h_new).  Returns (H,W).
    """
    h = np.asarray(h, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    H, W = h.shape
    ctx = gpu.ctx
    KN = "sw_continuity_conservative.comp"

    tex_h = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_h.write(h.tobytes())
    tex_u = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_u.write(u.tobytes())
    tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_v.write(v.tobytes())

    tex_s_pos = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_cap   = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_fx    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_fy    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_s_tot = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_hnew  = gpu.texture2d((W, H),     components=1, dtype="f4")

    gx_c = (W + _GROUP - 1) // _GROUP
    gy_c = (H + _GROUP - 1) // _GROUP
    gy_v = (H + 1 + _GROUP - 1) // _GROUP

    def _common(k):
        _set(k, "u_size",    (W, H))
        _set(k, "u_dt",      float(dt))
        _set(k, "u_h_floor", float(h_floor))
        _set(k, "u_a",       float(a))
        tex_h.use(location=0); _set(k, "u_h", 0)
        tex_u.use(location=1); _set(k, "u_u", 1)
        tex_v.use(location=2); _set(k, "u_v", 2)

    # PASS 0: s_pos
    k0 = gpu.compute(_KERNELS, KN, defines={"PASS": "0"})
    _common(k0)
    tex_s_pos.bind_to_image(0, read=False, write=True)
    k0.run(gx_c, gy_c, 1); ctx.memory_barrier()

    # PASS 1: cap
    k1 = gpu.compute(_KERNELS, KN, defines={"PASS": "1"})
    _common(k1)
    tex_s_pos.use(location=3); _set(k1, "u_s_pos", 3)
    tex_cap.bind_to_image(0, read=False, write=True)
    k1.run(gx_c, gy_c, 1); ctx.memory_barrier()

    # PASS 2: Fx_tot, Fy_tot
    k2 = gpu.compute(_KERNELS, KN, defines={"PASS": "2"})
    _common(k2)
    tex_s_pos.use(location=3); _set(k2, "u_s_pos", 3)
    tex_cap.use(location=4);   _set(k2, "u_cap",   4)
    tex_fx.bind_to_image(0, read=False, write=True)
    tex_fy.bind_to_image(1, read=False, write=True)
    k2.run(gx_c, gy_v, 1); ctx.memory_barrier()

    # PASS 3: s_tot
    k3 = gpu.compute(_KERNELS, KN, defines={"PASS": "3"})
    _common(k3)
    tex_fx.use(location=3); _set(k3, "u_fx_tot", 3)
    tex_fy.use(location=4); _set(k3, "u_fy_tot", 4)
    tex_s_tot.bind_to_image(0, read=False, write=True)
    k3.run(gx_c, gy_c, 1); ctx.memory_barrier()

    # PASS 4: h_new
    k4 = gpu.compute(_KERNELS, KN, defines={"PASS": "4"})
    _common(k4)
    tex_fx.use(location=3);    _set(k4, "u_fx_tot", 3)
    tex_fy.use(location=4);    _set(k4, "u_fy_tot", 4)
    tex_s_tot.use(location=5); _set(k4, "u_s_tot",  5)
    tex_hnew.bind_to_image(0, read=False, write=True)
    k4.run(gx_c, gy_c, 1); ctx.memory_barrier()

    result = gpu.read_texture(tex_hnew)[..., 0]
    for tex in (tex_h, tex_u, tex_v, tex_s_pos, tex_cap,
                tex_fx, tex_fy, tex_s_tot, tex_hnew):
        tex.release()
    return result


def run_helmholtz_rhs(
    gpu: "GpuContext",
    h_n: np.ndarray,
    u_n: np.ndarray,
    v_n: np.ndarray,
    u_star: np.ndarray,
    v_star: np.ndarray,
    dh_prev: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    omega: float,
    theta: float,
    dt: float,
    a: float,
) -> np.ndarray:
    """GPU Helmholtz RHS assembly; ports helmholtz_rhs() exactly, shape (H, W)."""
    h_n = np.asarray(h_n, dtype=np.float32)
    u_n = np.asarray(u_n, dtype=np.float32)
    v_n = np.asarray(v_n, dtype=np.float32)
    u_star = np.asarray(u_star, dtype=np.float32)
    v_star = np.asarray(v_star, dtype=np.float32)
    dh_prev = np.asarray(dh_prev, dtype=np.float32)
    H, W = h_n.shape
    ctx = gpu.ctx

    # grad(h_n), grad(dh_prev) on faces (gp=1 so run_grad returns the bare grad).
    gxn, gyn = run_grad(gpu, h_n, gp=1.0, a=a)
    gxd, gyd = run_grad(gpu, dh_prev, gp=1.0, a=a)

    tex_us  = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_us.write(u_star.tobytes())
    tex_vs  = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_vs.write(v_star.tobytes())
    tex_un  = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_un.write(u_n.tobytes())
    tex_vn  = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_vn.write(v_n.tobytes())
    tex_gxn = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_gxn.write(gxn.astype(np.float32).tobytes())
    tex_gyn = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_gyn.write(gyn.astype(np.float32).tobytes())
    tex_gxd = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_gxd.write(gxd.astype(np.float32).tobytes())
    tex_gyd = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_gyd.write(gyd.astype(np.float32).tobytes())
    tex_href = _upload_href(gpu, H_ref_lat, H)
    tex_out  = gpu.texture2d((W, H), components=1, dtype="f4")

    k = gpu.compute(_KERNELS, "sw_helmholtz_rhs.comp")
    _set(k, "u_size",  (W, H))
    _set(k, "u_a",     float(a))
    _set(k, "u_dlam",  2.0 * math.pi / W)
    _set(k, "u_dphi",  math.pi / H)
    _set(k, "u_omega", float(omega))
    _set(k, "u_dt",    float(dt))
    _set(k, "u_gp",    float(gp))
    _set(k, "u_theta", float(theta))
    tex_href.use(location=0); _set(k, "u_Href", 0)
    tex_us.use(location=1);   _set(k, "u_us",  1)
    tex_vs.use(location=2);   _set(k, "u_vs",  2)
    tex_un.use(location=3);   _set(k, "u_un",  3)
    tex_vn.use(location=4);   _set(k, "u_vn",  4)
    tex_gxn.use(location=5);  _set(k, "u_gxn", 5)
    tex_gyn.use(location=6);  _set(k, "u_gyn", 6)
    tex_gxd.use(location=7);  _set(k, "u_gxd", 7)
    tex_gyd.use(location=8);  _set(k, "u_gyd", 8)
    tex_out.bind_to_image(0, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    result = gpu.read_texture(tex_out)[..., 0]
    for tex in (tex_us, tex_vs, tex_un, tex_vn, tex_gxn, tex_gyn,
                tex_gxd, tex_gyd, tex_href, tex_out):
        tex.release()
    return result


def run_velocity_backsub(
    gpu: "GpuContext",
    u_star: np.ndarray,
    v_star: np.ndarray,
    h_impl: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    omega: float,
    a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU velocity back-substitution; ports velocity_backsub() exactly.

    Returns (u_new (H,W), v_new (H+1,W)).
    """
    u_star = np.asarray(u_star, dtype=np.float32)
    v_star = np.asarray(v_star, dtype=np.float32)
    h_impl = np.asarray(h_impl, dtype=np.float32)
    H, W = h_impl.shape
    ctx = gpu.ctx

    grad_x, grad_y = run_grad(gpu, h_impl, gp=1.0, a=a)

    tex_us = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_us.write(u_star.tobytes())
    tex_vs = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_vs.write(v_star.tobytes())
    tex_gx = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_gx.write(grad_x.astype(np.float32).tobytes())
    tex_gy = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_gy.write(grad_y.astype(np.float32).tobytes())
    tex_u  = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v  = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    k = gpu.compute(_KERNELS, "sw_velocity_backsub.comp")
    _set(k, "u_size",  (W, H))
    _set(k, "u_omega", float(omega))
    _set(k, "u_dt",    float(dt))
    _set(k, "u_gp",    float(gp))
    _set(k, "u_theta", float(theta))
    tex_us.use(location=0); _set(k, "u_us", 0)
    tex_vs.use(location=1); _set(k, "u_vs", 1)
    tex_gx.use(location=2); _set(k, "u_gx", 2)
    tex_gy.use(location=3); _set(k, "u_gy", 3)
    tex_u.bind_to_image(0, read=False, write=True)
    tex_v.bind_to_image(1, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    u_new = gpu.read_texture(tex_u)[..., 0]
    v_new = gpu.read_texture(tex_v)[..., 0]
    for tex in (tex_us, tex_vs, tex_gx, tex_gy, tex_u, tex_v):
        tex.release()
    return u_new, v_new


def run_si_predictor(
    gpu: "GpuContext",
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    gp: float,
    dt: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU semi-implicit explicit predictor; returns (u_star, v_star).

    Ports _semi_implicit_predictor() from shallow_water_ref.py exactly:
    advection (relative-vorticity flux) + KE-gradient + (1-theta) explicit
    pressure half -(1-theta)*dt*gp*grad(h^n).  NO Coriolis.

    Orchestration:
      1. zeta  = run_vorticity(u, v, a)                  — corners (H+1, W)
      2. ke    = 0.5*(u^2 + v_c^2) at centres (CPU-free) via numpy on GPU read?
         We build ke on CPU from the already-resident arrays (h,u,v are inputs).
      3. (gxk, gyk) = run_grad(ke, gp=1, a)
      4. (gxn, gyn) = run_grad(h,  gp=1, a)
      5. sw_si_predictor.comp combiner.
    """
    h = np.asarray(h, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    H, W = h.shape
    ctx = gpu.ctx

    zeta = run_vorticity(gpu, u, v, a)                # (H+1, W)
    v_c = 0.5 * (v[0:H] + v[1:H + 1])
    ke = (0.5 * (u * u + v_c * v_c)).astype(np.float32)
    gxk, gyk = run_grad(gpu, ke, gp=1.0, a=a)
    gxn, gyn = run_grad(gpu, h,  gp=1.0, a=a)

    tex_zeta = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_gxk  = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_gyk  = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_gxn  = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_gyn  = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_u    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_us   = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_vs   = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    tex_zeta.write(zeta.astype(np.float32).tobytes())
    tex_gxk.write(gxk.astype(np.float32).tobytes())
    tex_gyk.write(gyk.astype(np.float32).tobytes())
    tex_gxn.write(gxn.astype(np.float32).tobytes())
    tex_gyn.write(gyn.astype(np.float32).tobytes())
    tex_u.write(u.tobytes())
    tex_v.write(v.tobytes())

    k = gpu.compute(_KERNELS, "sw_si_predictor.comp")
    _set(k, "u_size",  (W, H))
    _set(k, "u_dt",    float(dt))
    _set(k, "u_gp",    float(gp))
    _set(k, "u_theta", float(theta))
    tex_zeta.use(location=0); _set(k, "u_zeta", 0)
    tex_gxk.use(location=1);  _set(k, "u_gxk",  1)
    tex_gyk.use(location=2);  _set(k, "u_gyk",  2)
    tex_gxn.use(location=3);  _set(k, "u_gxn",  3)
    tex_gyn.use(location=4);  _set(k, "u_gyn",  4)
    tex_u.use(location=5);    _set(k, "u_u",    5)
    tex_v.use(location=6);    _set(k, "u_v",    6)
    tex_us.bind_to_image(0, read=False, write=True)
    tex_vs.bind_to_image(1, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    u_star = gpu.read_texture(tex_us)[..., 0]
    v_star = gpu.read_texture(tex_vs)[..., 0]
    for tex in (tex_zeta, tex_gxk, tex_gyk, tex_gxn, tex_gyn,
                tex_u, tex_v, tex_us, tex_vs):
        tex.release()
    return u_star, v_star


def run_momentum(
    gpu: "GpuContext",
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    gp: float,
    omega: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU vector-invariant momentum step; returns (u_new, v_new).

    Orchestration:
      1. ζ = run_vorticity(gpu, u, v, a)               — corners (H+1, W)
      2. B = sw_bernoulli.comp(h, u, v, gp)             — centres (H, W)
      3. (gx, gy) = run_grad(gpu, B, gp=1.0, a)         — Bernoulli gradient (1 a-site)
      4. sw_momentum.comp(ζ, gx, gy, u, v, omega, dt)   — assembly (no metric here)

    Parameters
    ----------
    gpu   : GpuContext
    h     : (H, W) float32 — layer thickness
    u     : (H, W) float32 — zonal velocity at u-faces
    v     : (H+1, W) float32 — meridional velocity at v-faces
    a     : float — planetary radius
    gp    : float — reduced gravity g'
    omega : float — planetary rotation rate Ω (f = 2Ω sinφ)
    dt    : float — timestep

    Returns
    -------
    (u_new, v_new) — (H, W) and (H+1, W) float32 arrays.
    """
    h = np.asarray(h, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    H, W = h.shape
    ctx = gpu.ctx

    # --- Step 1: relative vorticity at corners (H+1, W) ---
    zeta = run_vorticity(gpu, u, v, a)  # numpy (H+1, W)

    # --- Step 2: Bernoulli potential B at centres (H, W) via GPU kernel ---
    tex_h_b = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_u_b = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v_b = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_B   = gpu.texture2d((W, H),     components=1, dtype="f4")

    tex_h_b.write(h.tobytes())
    tex_u_b.write(u.tobytes())
    tex_v_b.write(v.tobytes())

    kb = gpu.compute(_KERNELS, "sw_bernoulli.comp")
    _set(kb, "u_size", (W, H))
    _set(kb, "u_gp",   float(gp))
    tex_h_b.use(location=0); _set(kb, "u_h", 0)
    tex_u_b.use(location=1); _set(kb, "u_u", 1)
    tex_v_b.use(location=2); _set(kb, "u_v", 2)
    tex_B.bind_to_image(0, read=False, write=True)

    gx_grp = (W + _GROUP - 1) // _GROUP
    gy_grp = (H + _GROUP - 1) // _GROUP
    kb.run(gx_grp, gy_grp, 1)
    ctx.memory_barrier()

    B_arr = gpu.read_texture(tex_B)[..., 0]  # (H, W) float32

    for tex in (tex_h_b, tex_u_b, tex_v_b, tex_B):
        tex.release()

    # --- Step 3: Bernoulli gradient (gx, gy) via run_grad with gp=1.0 ---
    # B already contains gp*h + ke; pass gp=1.0 so run_grad computes grad(1.0*B)=grad(B).
    B_gx, B_gy = run_grad(gpu, B_arr, gp=1.0, a=a)  # (H,W) and (H+1,W)

    # --- Step 4: assembly kernel ---
    tex_zeta  = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_gx    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_gy    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_u_m   = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v_m   = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_u_new = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v_new = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    tex_zeta.write(zeta.astype(np.float32).tobytes())
    tex_gx.write(B_gx.astype(np.float32).tobytes())
    tex_gy.write(B_gy.astype(np.float32).tobytes())
    tex_u_m.write(u.tobytes())
    tex_v_m.write(v.tobytes())

    km = gpu.compute(_KERNELS, "sw_momentum.comp")
    _set(km, "u_size",  (W, H))
    _set(km, "u_omega", float(omega))
    _set(km, "u_dt",    float(dt))

    tex_zeta.use(location=0); _set(km, "u_zeta", 0)
    tex_gx.use(location=1);   _set(km, "u_gx",   1)
    tex_gy.use(location=2);   _set(km, "u_gy",   2)
    tex_u_m.use(location=3);  _set(km, "u_u",    3)
    tex_v_m.use(location=4);  _set(km, "u_v",    4)

    tex_u_new.bind_to_image(0, read=False, write=True)
    tex_v_new.bind_to_image(1, read=False, write=True)

    gx_grp = (W + _GROUP - 1) // _GROUP
    gy_grp = (H + 1 + _GROUP - 1) // _GROUP
    km.run(gx_grp, gy_grp, 1)
    ctx.memory_barrier()

    result_u = gpu.read_texture(tex_u_new)[..., 0]  # (H, W)
    result_v = gpu.read_texture(tex_v_new)[..., 0]  # (H+1, W)

    for tex in (tex_zeta, tex_gx, tex_gy, tex_u_m, tex_v_m, tex_u_new, tex_v_new):
        tex.release()

    return result_u, result_v


# ===========================================================================
# M3-T8: GPU 2-layer port — Montgomery / Bernoulli-M / per-layer momentum /
# forcing orchestrators (a-aware production, diffed per-field vs the CPU ref).
# ===========================================================================

def run_montgomery(
    gpu: "GpuContext",
    h1: np.ndarray,
    h2: np.ndarray,
    gp1: float,
    gp2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU 2-layer Montgomery potentials (M1, M2), each (H, W).

    Ports montgomery_2layer(): M1 = gp1*(h1+h2), M2 = gp1*(h1+h2)+gp2*h2.
    A potential — a-agnostic (no metric here).
    """
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    H, W = h1.shape
    ctx = gpu.ctx

    tex_h1 = gpu.texture2d((W, H), components=1, dtype="f4"); tex_h1.write(h1.tobytes())
    tex_h2 = gpu.texture2d((W, H), components=1, dtype="f4"); tex_h2.write(h2.tobytes())
    tex_M1 = gpu.texture2d((W, H), components=1, dtype="f4")
    tex_M2 = gpu.texture2d((W, H), components=1, dtype="f4")

    k = gpu.compute(_KERNELS, "sw_montgomery.comp")
    _set(k, "u_size", (W, H))
    _set(k, "u_gp1",  float(gp1))
    _set(k, "u_gp2",  float(gp2))
    tex_h1.use(location=0); _set(k, "u_h1", 0)
    tex_h2.use(location=1); _set(k, "u_h2", 1)
    tex_M1.bind_to_image(0, read=False, write=True)
    tex_M2.bind_to_image(1, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    M1 = gpu.read_texture(tex_M1)[..., 0]
    M2 = gpu.read_texture(tex_M2)[..., 0]
    for tex in (tex_h1, tex_h2, tex_M1, tex_M2):
        tex.release()
    return M1, M2


def run_bernoulli_M(
    gpu: "GpuContext",
    M: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """GPU Bernoulli-M stage: B = M + 0.5*(u^2 + v_c^2), shape (H, W).

    Uses the USE_M variant of sw_bernoulli.comp (the M3 pressure seam).
    """
    M = np.asarray(M, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    H, W = M.shape
    ctx = gpu.ctx

    tex_M = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_M.write(M.tobytes())
    tex_u = gpu.texture2d((W, H),     components=1, dtype="f4"); tex_u.write(u.tobytes())
    tex_v = gpu.texture2d((W, H + 1), components=1, dtype="f4"); tex_v.write(v.tobytes())
    tex_B = gpu.texture2d((W, H),     components=1, dtype="f4")

    k = gpu.compute(_KERNELS, "sw_bernoulli.comp", defines={"USE_M": "1"})
    _set(k, "u_size", (W, H))
    tex_M.use(location=0); _set(k, "u_M", 0)
    tex_u.use(location=1); _set(k, "u_u", 1)
    tex_v.use(location=2); _set(k, "u_v", 2)
    tex_B.bind_to_image(0, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + _GROUP - 1) // _GROUP
    k.run(gx, gy, 1)
    ctx.memory_barrier()

    B = gpu.read_texture(tex_B)[..., 0]
    for tex in (tex_M, tex_u, tex_v, tex_B):
        tex.release()
    return B


def run_momentum_M(
    gpu: "GpuContext",
    M: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    omega: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU Montgomery-driven momentum step for ONE layer; ports momentum_step_M.

    Chain: vorticity(u,v,a) -> bernoulli_M(M,u,v) -> grad(B,gp=1,a) -> momentum.
    Mirrors the production single-layer momentum chain with B = M + ke instead of
    B = gp*h + ke.  Returns (u_new (H,W), v_new (H+1,W)).
    """
    M = np.asarray(M, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    H, W = M.shape
    ctx = gpu.ctx

    zeta = run_vorticity(gpu, u, v, a)             # (H+1, W)
    B = run_bernoulli_M(gpu, M, u, v)              # (H, W)
    B_gx, B_gy = run_grad(gpu, B, gp=1.0, a=a)     # grad(B)

    tex_zeta  = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_gx    = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_gy    = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_u_m   = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v_m   = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    tex_u_new = gpu.texture2d((W, H),     components=1, dtype="f4")
    tex_v_new = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    tex_zeta.write(zeta.astype(np.float32).tobytes())
    tex_gx.write(B_gx.astype(np.float32).tobytes())
    tex_gy.write(B_gy.astype(np.float32).tobytes())
    tex_u_m.write(u.tobytes())
    tex_v_m.write(v.tobytes())

    km = gpu.compute(_KERNELS, "sw_momentum.comp")
    _set(km, "u_size",  (W, H))
    _set(km, "u_omega", float(omega))
    _set(km, "u_dt",    float(dt))
    tex_zeta.use(location=0); _set(km, "u_zeta", 0)
    tex_gx.use(location=1);   _set(km, "u_gx",   1)
    tex_gy.use(location=2);   _set(km, "u_gy",   2)
    tex_u_m.use(location=3);  _set(km, "u_u",    3)
    tex_v_m.use(location=4);  _set(km, "u_v",    4)
    tex_u_new.bind_to_image(0, read=False, write=True)
    tex_v_new.bind_to_image(1, read=False, write=True)

    gx = (W + _GROUP - 1) // _GROUP
    gy = (H + 1 + _GROUP - 1) // _GROUP
    km.run(gx, gy, 1)
    ctx.memory_barrier()

    u_new = gpu.read_texture(tex_u_new)[..., 0]
    v_new = gpu.read_texture(tex_v_new)[..., 0]
    for tex in (tex_zeta, tex_gx, tex_gy, tex_u_m, tex_v_m, tex_u_new, tex_v_new):
        tex.release()
    return u_new, v_new


def run_forcing_2layer(gpu: "GpuContext", st) -> dict[str, np.ndarray]:
    """GPU 2-layer forcing pass; ports shallow_water_ref.apply_forcing().

    Two passes of sw_forcing.comp (relax+drag -> hypervisc+sponge+floor).
    `st` is a Sw2State (reads h1/u1/v1/h2/u2/v2, h_eq1/h_eq2, tau_rad/tau_drag/
    nu4/sponge_rate/h_floor).  Returns a dict of post-forcing fields; does NOT
    mutate st.
    """
    g = st.g
    H, W = g.H, g.W
    ctx = gpu.ctx

    def cc(arr):
        t = gpu.texture2d((W, H), components=1, dtype="f4")
        t.write(np.asarray(arr, dtype=np.float32).tobytes()); return t

    def vf(arr):
        t = gpu.texture2d((W, H + 1), components=1, dtype="f4")
        t.write(np.asarray(arr, dtype=np.float32).tobytes()); return t

    heq1 = st.h_eq1 if st.h_eq1 is not None else np.zeros((H, W), np.float32)
    heq2 = st.h_eq2 if st.h_eq2 is not None else np.zeros((H, W), np.float32)

    tex_h1 = cc(st.h1); tex_h2 = cc(st.h2)
    tex_u1 = cc(st.u1); tex_u2 = cc(st.u2)
    tex_v1 = vf(st.v1); tex_v2 = vf(st.v2)
    tex_heq1 = cc(heq1); tex_heq2 = cc(heq2)

    sc_h1 = gpu.texture2d((W, H), components=1, dtype="f4")
    sc_h2 = gpu.texture2d((W, H), components=1, dtype="f4")
    sc_u1 = gpu.texture2d((W, H), components=1, dtype="f4")
    sc_u2 = gpu.texture2d((W, H), components=1, dtype="f4")
    sc_v1 = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    sc_v2 = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    out_h1 = gpu.texture2d((W, H), components=1, dtype="f4")
    out_h2 = gpu.texture2d((W, H), components=1, dtype="f4")
    out_u1 = gpu.texture2d((W, H), components=1, dtype="f4")
    out_u2 = gpu.texture2d((W, H), components=1, dtype="f4")
    out_v1 = gpu.texture2d((W, H + 1), components=1, dtype="f4")
    out_v2 = gpu.texture2d((W, H + 1), components=1, dtype="f4")

    gx = (W + _GROUP - 1) // _GROUP
    gy_v = (H + 1 + _GROUP - 1) // _GROUP

    def _common(k):
        _set(k, "u_size",        (W, H))
        _set(k, "u_tau_rad",     float(st.tau_rad))
        _set(k, "u_tau_drag",    float(st.tau_drag))
        _set(k, "u_nu4",         float(st.nu4))
        _set(k, "u_sponge_rate", float(st.sponge_rate))
        _set(k, "u_h_floor",     float(st.h_floor))

    # PASS 0
    k0 = gpu.compute(_KERNELS, "sw_forcing.comp", defines={"PASS0": "1"})
    _common(k0)
    tex_h1.use(location=0);   _set(k0, "u_h1", 0)
    tex_h2.use(location=1);   _set(k0, "u_h2", 1)
    tex_u1.use(location=2);   _set(k0, "u_u1", 2)
    tex_u2.use(location=3);   _set(k0, "u_u2", 3)
    tex_v1.use(location=4);   _set(k0, "u_v1", 4)
    tex_v2.use(location=5);   _set(k0, "u_v2", 5)
    tex_heq1.use(location=6); _set(k0, "u_h_eq1", 6)
    tex_heq2.use(location=7); _set(k0, "u_h_eq2", 7)
    sc_h1.bind_to_image(0, read=False, write=True)
    sc_h2.bind_to_image(1, read=False, write=True)
    sc_u1.bind_to_image(2, read=False, write=True)
    sc_u2.bind_to_image(3, read=False, write=True)
    sc_v1.bind_to_image(4, read=False, write=True)
    sc_v2.bind_to_image(5, read=False, write=True)
    k0.run(gx, gy_v, 1)
    ctx.memory_barrier()

    # PASS 1
    k1 = gpu.compute(_KERNELS, "sw_forcing.comp", defines={})
    _common(k1)
    tex_heq1.use(location=6); _set(k1, "u_h_eq1", 6)
    tex_heq2.use(location=7); _set(k1, "u_h_eq2", 7)
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
    k1.run(gx, gy_v, 1)
    ctx.memory_barrier()

    result = {
        "h1": gpu.read_texture(out_h1)[..., 0],
        "h2": gpu.read_texture(out_h2)[..., 0],
        "u1": gpu.read_texture(out_u1)[..., 0],
        "u2": gpu.read_texture(out_u2)[..., 0],
        "v1": gpu.read_texture(out_v1)[..., 0],
        "v2": gpu.read_texture(out_v2)[..., 0],
    }
    for tex in (tex_h1, tex_h2, tex_u1, tex_u2, tex_v1, tex_v2, tex_heq1, tex_heq2,
                sc_h1, sc_h2, sc_u1, sc_u2, sc_v1, sc_v2,
                out_h1, out_h2, out_u1, out_u2, out_v1, out_v2):
        tex.release()
    return result

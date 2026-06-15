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
    ) -> "SwGpuSolver":
        """Build a SwGpuSolver from the analytic Williamson-2 initial condition.

        Produces the same initial h, u, v and dt as
        ``ref.williamson2_state(W, H, a, omega, u0, gp, h0)``.
        """
        from gasgiant.sim import shallow_water_ref as ref  # noqa: PLC0415
        st = ref.williamson2_state(W=W, H=H, a=a, omega=omega, u0=u0, gp=gp,
                                   h0=h0, h_floor=h_floor)
        sg = cls(gpu=gpu, W=W, H=H, a=a, gp=gp, omega=omega,
                 dt=st.dt, h_floor=h_floor)
        sg._tex_h.write(st.h.astype(np.float32).tobytes())
        sg._tex_u.write(st.u.astype(np.float32).tobytes())
        sg._tex_v.write(st.v.astype(np.float32).tobytes())
        # Store initial velocity fields for velocity_l2_drift().
        sg.u_init = st.u.astype(np.float32).copy()
        sg.v_init = st.v.astype(np.float32).copy()
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

    # -- Step -----------------------------------------------------------------

    def step(self) -> None:
        """One full step in CPU-reference order (no CPU round-trip).

        Sequence (mirrors ref.step / ref.momentum_step order):
          1. vorticity(old u, v) → ζ
          2. bernoulli(old h, u, v) → B
          3. grad(B, gp=1.0) → gx, gy
          4. momentum-assembly(ζ, gx, gy, old u, v) → u_new, v_new
          5. continuity pass-A (old h, u_new, v_new) → h_low, cap
          6. continuity pass-B → h_new
          7. ping-pong: swap resident h/u/v ↔ h_new/u_new/v_new
        """
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
        k.run(gx_c, gy_c, 1)
        ctx.memory_barrier()

        # --- 7. Ping-pong: swap resident ↔ new ------------------------------
        self._tex_h, self._tex_h_new = self._tex_h_new, self._tex_h
        self._tex_u, self._tex_u_new = self._tex_u_new, self._tex_u
        self._tex_v, self._tex_v_new = self._tex_v_new, self._tex_v


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

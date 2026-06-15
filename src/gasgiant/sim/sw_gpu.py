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

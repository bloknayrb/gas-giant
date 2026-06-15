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

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import moderngl
    from gasgiant.gl.context import GpuContext


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

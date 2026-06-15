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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import moderngl
    from gasgiant.gl.context import GpuContext


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

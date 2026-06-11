"""Tracer state: one RGBA32F texture holds all four tracers
(r=T0 color index, g=T1 height, b=T2 detail, a=T3 storm tint).

MacCormack needs the current, forward, and back-and-forth fields live at once,
plus a separate write target (a sampler and an image binding may not alias the
same texture), so the state owns four buffers in rotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext

if TYPE_CHECKING:
    import moderngl


class TracerState:
    def __init__(self, gpu: GpuContext, size: tuple[int, int]) -> None:
        self.gpu = gpu
        self.size = size
        # linear=True: the derive pass samples with texture() (one-shot, fine);
        # the sim loop uses texelFetch, which ignores the filter state.
        self.cur: moderngl.Texture = gpu.texture2d(size, 4, "f4", linear=True)
        self.fwd: moderngl.Texture = gpu.texture2d(size, 4, "f4", linear=True)
        self.back: moderngl.Texture = gpu.texture2d(size, 4, "f4", linear=True)
        self.out: moderngl.Texture = gpu.texture2d(size, 4, "f4", linear=True)

    def commit(self) -> None:
        """After the correct pass wrote into ``out``, make it current."""
        self.cur, self.out = self.out, self.cur

    def read_current(self) -> np.ndarray:
        return self.gpu.read_texture(self.cur)

    def release(self) -> None:
        for tex in (self.cur, self.fwd, self.back, self.out):
            tex.release()

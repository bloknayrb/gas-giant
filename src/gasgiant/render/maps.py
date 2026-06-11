"""Map derivation: tracer state -> color/height textures at any resolution.
One kernel serves the live preview and export."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext
from gasgiant.palette import bake_lut, bake_rows
from gasgiant.params.model import AppearanceParams, GradientStop

if TYPE_CHECKING:
    import moderngl

_KERNELS = "gasgiant.render.kernels"
_GROUP = 16


def _stops(stops: list[GradientStop]) -> list[tuple[float, tuple[float, float, float]]]:
    return [(s.pos, s.color) for s in stops]


class MapDeriver:
    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        self.prog = gpu.compute(_KERNELS, "derive.comp")
        self._palette_tex: moderngl.Texture | None = None
        self._storm_tex: moderngl.Texture | None = None

    def update_palettes(self, appearance: AppearanceParams) -> None:
        for tex in (self._palette_tex, self._storm_tex):
            if tex is not None:
                tex.release()
        rows = [(row.latitude, _stops(row.stops)) for row in appearance.palette_rows]
        self._palette_tex = self.gpu.lut_texture(bake_rows(rows, height=64))
        self._storm_tex = self.gpu.lut_texture(bake_lut(_stops(appearance.storm_tints)))

    def derive(
        self,
        tracers: moderngl.Texture,
        patch_n: moderngl.Texture,
        patch_s: moderngl.Texture,
        patch_rho_max: float,
        blend_band: tuple[float, float],
        color_out: moderngl.Texture,
        height_out: moderngl.Texture,
        appearance: AppearanceParams,
        detail_gain: float = 0.35,
        detail_tex: moderngl.Texture | None = None,
        detail_intensity: float = 0.0,
        origin: tuple[int, int] = (0, 0),
        full_size: tuple[int, int] | None = None,
        lanes: list[tuple[float, float]] | None = None,
        warp: tuple[tuple[float, float, float], float, float] | None = None,
    ) -> None:
        """lanes: (latitude, strength) thin dark lane lines; warp: the band
        meander (offset, amount, freq) the lanes ride on."""
        if self._palette_tex is None:
            self.update_palettes(appearance)
        prog = self.prog
        size = color_out.size
        lanes = lanes or []
        packed = np.zeros((16, 2), dtype=np.float32)
        for i, lane in enumerate(lanes[:16]):
            packed[i] = lane
        prog["u_lane_count"].value = min(len(lanes), 16)
        prog["u_lanes"].write(packed.tobytes())
        w_off, w_amount, w_freq = warp if warp is not None else ((0.0, 0.0, 0.0), 0.0, 3.0)
        prog["u_warp_offset"].value = w_off
        prog["u_warp_amount"].value = w_amount
        prog["u_warp_freq"].value = w_freq
        prog["u_origin"].value = origin
        prog["u_full_size"].value = full_size if full_size is not None else size
        tracers.use(location=0)
        prog["u_tracers"].value = 0
        self._palette_tex.use(location=1)
        prog["u_palette"].value = 1
        self._storm_tex.use(location=2)
        prog["u_storm_palette"].value = 2
        patch_n.use(location=3)
        prog["u_patch_n"].value = 3
        patch_s.use(location=4)
        prog["u_patch_s"].value = 4
        prog["u_patch_rho_max"].value = patch_rho_max
        prog["u_blend_lo"].value = blend_band[0]
        prog["u_blend_hi"].value = blend_band[1]
        (detail_tex if detail_tex is not None else self._palette_tex).use(location=5)
        prog["u_detail"].value = 5
        prog["u_detail_intensity"].value = detail_intensity if detail_tex is not None else 0.0
        prog["u_size"].value = size
        prog["u_detail_gain"].value = detail_gain
        prog["u_haze_amount"].value = appearance.haze_amount
        prog["u_haze_color"].value = appearance.haze_color
        prog["u_contrast"].value = appearance.contrast
        prog["u_saturation"].value = appearance.saturation
        prog["u_gamma"].value = appearance.gamma
        color_out.bind_to_image(0, read=False, write=True)
        height_out.bind_to_image(1, read=False, write=True)
        gx = (size[0] + _GROUP - 1) // _GROUP
        gy = (size[1] + _GROUP - 1) // _GROUP
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

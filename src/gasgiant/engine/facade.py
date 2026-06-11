"""The Simulation facade.

Phase 1: a one-shot render of the spine kernel (banded sphere-noise planet)
at any resolution. The facade boundary is what survives: the GUI, CLI, and
tests already consume only this; the advected-tracer pipeline replaces the
internals in Phase 3 (init/step/preview/export, invalidation dispatch,
snapshotting).
"""

from __future__ import annotations

import logging

import numpy as np

from gasgiant.core.domain import EquirectGrid
from gasgiant.gl import GpuContext
from gasgiant.palette import bake_lut
from gasgiant.params.model import MAX_BANDS, GradientStop, PlanetParams
from gasgiant.params.seeds import subseed
from gasgiant.sim.bands import generate_bands

log = logging.getLogger(__name__)

_KERNEL_PACKAGE = "gasgiant.sim.kernels"
_GROUP = 16


def _stops_to_tuples(stops: list[GradientStop]) -> list[tuple[float, tuple[float, float, float]]]:
    return [(s.pos, s.color) for s in stops]


class Simulation:
    def __init__(self, params: PlanetParams, gpu: GpuContext | None = None) -> None:
        self.params = params
        self.gpu = gpu if gpu is not None else GpuContext.headless()
        self._spine = self.gpu.compute(_KERNEL_PACKAGE, "spine.comp")

    def render_maps(self, width: int | None = None) -> dict[str, np.ndarray]:
        """Render all maps at the given equirect width (default: export width).

        Returns {"color": (H, W, 4) float32 0..1, "height": (H, W) float32 0..1}.
        """
        p = self.params
        grid = EquirectGrid(width or p.export.width, (width or p.export.width) // 2)
        ctx = self.gpu.ctx

        color_tex = self.gpu.texture2d(grid.size, components=4, dtype="f4")
        height_tex = self.gpu.texture2d(grid.size, components=1, dtype="f4")

        lut = bake_lut(_stops_to_tuples(p.appearance.palette))
        lut_tex = ctx.texture((lut.shape[0], 1), 4, data=lut.tobytes(), dtype="f4")
        lut_tex.repeat_x = False

        bands = generate_bands(p.seed, p.bands)
        warp_rng = subseed(p.seed, "warp-noise")
        detail_rng = subseed(p.seed, "detail-noise")

        k = self._spine
        k["u_size"].value = grid.size
        k["u_band_count"].value = len(bands.values)
        k["u_band_edges"].write(_pad(bands.edges, MAX_BANDS + 1))
        k["u_band_values"].write(_pad(bands.values, MAX_BANDS))
        k["u_band_heights"].write(_pad(bands.heights, MAX_BANDS))
        k["u_edge_softness"].value = p.bands.edge_softness
        k["u_noise_offset_warp"].value = tuple(warp_rng.uniform(-100.0, 100.0, 3))
        k["u_noise_offset_detail"].value = tuple(detail_rng.uniform(-100.0, 100.0, 3))
        k["u_warp_amount"].value = p.bands.warp_amount
        k["u_warp_freq"].value = p.bands.warp_freq
        k["u_detail_amount"].value = p.bands.detail_amount
        k["u_detail_freq"].value = p.bands.detail_freq
        k["u_haze_amount"].value = p.appearance.haze_amount
        k["u_haze_color"].value = p.appearance.haze_color
        k["u_contrast"].value = p.appearance.contrast
        k["u_saturation"].value = p.appearance.saturation
        k["u_gamma"].value = p.appearance.gamma

        color_tex.bind_to_image(0, read=False, write=True)
        height_tex.bind_to_image(1, read=False, write=True)
        lut_tex.use(location=0)
        k["u_palette"].value = 0

        groups_x = (grid.width + _GROUP - 1) // _GROUP
        groups_y = (grid.height + _GROUP - 1) // _GROUP
        k.run(groups_x, groups_y, 1)
        ctx.memory_barrier()

        color = self.gpu.read_texture(color_tex)
        height = self.gpu.read_texture(height_tex)[..., 0]

        for tex in (color_tex, height_tex, lut_tex):
            tex.release()

        return {"color": color, "height": height}


def _pad(arr: np.ndarray, size: int) -> bytes:
    out = np.zeros(size, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out.tobytes()

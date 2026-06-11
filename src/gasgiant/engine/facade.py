"""The Simulation facade.

Phase 2 shape: the facade owns persistent preview textures (zero-copy display
in the GUI), accepts validated parameter updates via the invalidation
machinery, and still does one-shot full-resolution renders for export.
The advected-tracer pipeline replaces the internals in Phase 3; this boundary
(update_params / ensure_preview / render_maps) is what survives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type names only — GL object creation stays under gl/
    import moderngl

from gasgiant.core.domain import EquirectGrid
from gasgiant.engine.invalidation import diff_tiers
from gasgiant.gl import GpuContext
from gasgiant.palette import bake_lut
from gasgiant.params.model import MAX_BANDS, GradientStop, PlanetParams, Tier
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
        self._preview_color: moderngl.Texture | None = None
        self._preview_height: moderngl.Texture | None = None
        self._preview_dirty = True

    # -- parameters ----------------------------------------------------------

    def update_params(self, new_params: PlanetParams) -> set[Tier]:
        """Adopt new validated params; returns the invalidation tiers touched."""
        tiers = diff_tiers(self.params, new_params)
        self.params = new_params
        if tiers:
            # Phase 2: every tier re-renders the spine kernel (cheap). From
            # Phase 3 on, VELOCITY rebuilds psi and RESTART re-inits tracers.
            self._preview_dirty = True
        return tiers

    # -- preview (GPU-resident, no readback) ----------------------------------

    def ensure_preview(self, width: int) -> tuple[moderngl.Texture, bool]:
        """Render the preview if dirty or resized.

        Returns (color texture, was_rerendered). The texture stays owned by
        the engine and remains valid until the next resize.
        """
        height = width // 2
        recreated = False
        if self._preview_color is None or self._preview_color.size != (width, height):
            if self._preview_color is not None:
                self._preview_color.release()
                self._preview_height.release()
            self._preview_color = self.gpu.texture2d((width, height), components=4, dtype="f4")
            self._preview_height = self.gpu.texture2d((width, height), components=1, dtype="f4")
            self._preview_dirty = True
            recreated = True
        if self._preview_dirty:
            self._dispatch(EquirectGrid(width, height), self._preview_color, self._preview_height)
            self._preview_dirty = False
            return self._preview_color, True
        return self._preview_color, recreated

    @property
    def preview_height_texture(self) -> moderngl.Texture | None:
        return self._preview_height

    # -- export (full-res, readback) ------------------------------------------

    def render_maps(self, width: int | None = None) -> dict[str, np.ndarray]:
        """One-shot render + readback at the given equirect width.

        Returns {"color": (H, W, 4) float32 0..1, "height": (H, W) float32 0..1}.
        """
        w = width or self.params.export.width
        grid = EquirectGrid(w, w // 2)
        color_tex = self.gpu.texture2d(grid.size, components=4, dtype="f4")
        height_tex = self.gpu.texture2d(grid.size, components=1, dtype="f4")
        try:
            self._dispatch(grid, color_tex, height_tex)
            color = self.gpu.read_texture(color_tex)
            height = self.gpu.read_texture(height_tex)[..., 0]
        finally:
            color_tex.release()
            height_tex.release()
        return {"color": color, "height": height}

    # -- internals -------------------------------------------------------------

    def _dispatch(
        self,
        grid: EquirectGrid,
        color_tex: moderngl.Texture,
        height_tex: moderngl.Texture,
    ) -> None:
        p = self.params
        lut_tex = self.gpu.lut_texture(bake_lut(_stops_to_tuples(p.appearance.palette)))

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
        self.gpu.ctx.memory_barrier()
        lut_tex.release()


def _pad(arr: np.ndarray, size: int) -> bytes:
    out = np.zeros(size, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out.tobytes()

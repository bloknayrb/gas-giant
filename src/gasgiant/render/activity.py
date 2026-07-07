"""Activity pass: local strain + vorticity from the baked equirect velocity,
plus the CPU reduction (per-latitude-row mean strain + global eddy/vort means)
that the FIELD_DRIVE detail variant normalizes against.

The reduction is numpy, not a GPU tree reduction: it is exactly deterministic,
yields the per-row mean (for eddy strain = strain - rowmean) for free, and runs
identically in the preview and export paths so ``preview == export`` holds by
construction. It costs one texture readback per build; ``build`` is only invoked
where the caller already reads GPU textures (derive / per-export-frame), so the
single synchronous readback is not on any hot per-pixel path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext

if TYPE_CHECKING:
    import moderngl

_KERNELS = "gasgiant.render.kernels"
_GROUP = 16


@dataclass
class ActivityMeans:
    """Snapshot/caller-scoped reduction outputs. ``rowmean_tex`` is a sim-H x 1
    LUT (``.r`` = masked per-row mean strain) sampled by latitude in
    detail.comp; the caller owns it and must ``release()`` it when the paired
    activity texture is released (so the means are snapshot-scoped exactly like
    the activity texture -- no export tearing)."""

    mean_eddy: float
    mean_vort: float
    rowmean_tex: moderngl.Texture

    def release(self) -> None:
        self.rowmean_tex.release()


def new_activity_texture(gpu: GpuContext, size: tuple[int, int]) -> moderngl.Texture:
    """RG32F, sim-res activity target. Allocated with a plain LINEAR min-filter
    (level-0 complete) so the compute ``imageStore`` succeeds; ``build`` promotes
    it to a mipmap min-filter AFTER generating the mip chain, so the detail
    ``textureLod`` fill sample has complete mips. Setting the mipmap filter
    before any levels exist makes the texture mipmap-INCOMPLETE and the image
    store silently no-ops (readback returns uninitialized memory)."""
    return gpu.texture2d(size, 2, "f4", linear=True)


class ActivitySynth:
    SIM_MASK_DEG = 66.0  # |lat|<66 deg band for the means (aligned to ROUTE_LO)

    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        self.prog = gpu.compute(_KERNELS, "activity.comp")

    def build(
        self, vel_tex: moderngl.Texture, out_tex: moderngl.Texture
    ) -> ActivityMeans:
        """Fill ``out_tex`` (RG32F, sim res, mip-capable) with raw strain/vort,
        build its mipmaps, and reduce to the per-row + global means."""
        w, h = out_tex.size
        vel_tex.use(location=0)
        self.prog["u_vel"].value = 0
        self.prog["u_size"].value = (w, h)
        self.prog["u_texel"].value = (1.0 / w, 1.0 / h)
        out_tex.bind_to_image(0, read=False, write=True)
        gx = (w + _GROUP - 1) // _GROUP
        gy = (h + _GROUP - 1) // _GROUP
        self.prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()
        # Build the mip chain, THEN promote to a mipmap min-filter, so the detail
        # textureLod fill sample has complete mips (see new_activity_texture).
        import moderngl

        out_tex.build_mipmaps()
        out_tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)

        # CPU reduction: deterministic, exact, per-row (for eddy strain).
        arr = self.gpu.read_texture(out_tex)            # (h, w, 2)
        strain = arr[:, :, 0]
        vort = np.abs(arr[:, :, 1])
        lat_deg = (0.5 - (np.arange(h) + 0.5) / h) * 180.0
        band = np.abs(lat_deg) < self.SIM_MASK_DEG      # (h,)
        rowmean = strain.mean(axis=1)                   # (h,) full row (all lon)
        eddy = np.clip(strain - rowmean[:, None], 0.0, None)
        mean_eddy = float(eddy[band].mean()) if band.any() else 0.0
        mean_vort = float(vort[band].mean()) if band.any() else 0.0
        lut = np.zeros((h, 4), dtype=np.float32)
        lut[:, 0] = rowmean
        rowmean_tex = self.gpu.lut_texture(lut)         # h x 1, linear, clamped
        return ActivityMeans(mean_eddy, mean_vort, rowmean_tex)

    def release(self) -> None:
        # The program lives in the GpuContext compute cache; nothing else owned.
        pass

"""Detail synthesis: flow-stretched filament noise + convective cells at the
output resolution, from the baked velocity and tracer textures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext
from gasgiant.params.model import DetailParams
from gasgiant.params.seeds import subseed

if TYPE_CHECKING:
    import moderngl

_KERNELS = "gasgiant.render.kernels"
_GROUP = 16


class DetailSynth:
    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        self.prog = gpu.compute(_KERNELS, "detail.comp")

    def synthesize(
        self,
        seed: int,
        vel_tex: moderngl.Texture,
        tracers_tex: moderngl.Texture,
        profile_dyn: moderngl.Texture,
        out_tex: moderngl.Texture,
        params: DetailParams,
        origin: tuple[int, int] = (0, 0),
        full_size: tuple[int, int] | None = None,
        heroes: list[tuple[float, float, float, float]] | None = None,
    ) -> None:
        """heroes: up to 3 (x, y, z, r_core) hero-storm centers; the detail
        amplitude and winding time grow inside them (internal spirals)."""
        rng = subseed(seed, "detail-synth")
        prog = self.prog
        size = out_tex.size
        prog["u_origin"].value = origin
        prog["u_full_size"].value = full_size if full_size is not None else size
        packed = np.zeros((3, 4), dtype=np.float32)
        n_heroes = 0
        for h in (heroes or [])[:3]:
            packed[n_heroes] = h
            n_heroes += 1
        prog["u_hero_count"].value = n_heroes
        prog["u_heroes"].write(packed.tobytes())
        vel_tex.use(location=0)
        prog["u_vel"].value = 0
        tracers_tex.use(location=1)
        prog["u_tracers"].value = 1
        profile_dyn.use(location=2)
        prog["u_profile_dyn"].value = 2
        prog["u_size"].value = size
        prog["u_freq"].value = params.frequency
        prog["u_stretch"].value = params.flow_stretch
        prog["u_phases"].value = params.flow_phases
        prog["u_cell_amount"].value = params.cellular_amount
        prog["u_offset"].value = tuple(rng.uniform(-100.0, 100.0, 3))
        out_tex.bind_to_image(0, read=False, write=True)
        gx = (size[0] + _GROUP - 1) // _GROUP
        gy = (size[1] + _GROUP - 1) // _GROUP
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

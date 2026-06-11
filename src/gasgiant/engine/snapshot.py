"""Export snapshots: an immutable copy of everything a tiled export reads.

The export spans many GUI frames; without a snapshot, mid-export slider drags
or sim stepping would make tiles 3 and 47 disagree (invisible until you get
hard tile seams — the design review's correctness blocker). All copies are
GPU-side blits; params and derived state are deep-copied on the CPU. This is
also the mechanism a future animation exporter steps through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gasgiant.params.model import PlanetParams

if TYPE_CHECKING:
    import moderngl

    from gasgiant.gl import GpuContext


@dataclass
class ExportSnapshot:
    params: PlanetParams
    tracers_eq: moderngl.Texture
    tracers_n: moderngl.Texture
    tracers_s: moderngl.Texture
    vel_eq: moderngl.Texture
    profile_dyn: moderngl.Texture
    patch_rho_max: float
    blend_band: tuple[float, float]

    @classmethod
    def capture(cls, sim) -> ExportSnapshot:  # sim: engine.Simulation
        gpu: GpuContext = sim.gpu
        s = sim.solver
        from gasgiant.sim.solver import BLEND_BAND, RHO_MAX

        return cls(
            params=sim.params.model_copy(deep=True),
            tracers_eq=gpu.clone_texture(s.equirect.tracers.cur),
            tracers_n=gpu.clone_texture(s.north.tracers.cur),
            tracers_s=gpu.clone_texture(s.south.tracers.cur),
            vel_eq=gpu.clone_texture(s.equirect.vel_tex),
            profile_dyn=gpu.clone_texture(sim.profile_dyn),
            patch_rho_max=RHO_MAX,
            blend_band=BLEND_BAND,
        )

    def release(self) -> None:
        for tex in (self.tracers_eq, self.tracers_n, self.tracers_s,
                    self.vel_eq, self.profile_dyn):
            tex.release()

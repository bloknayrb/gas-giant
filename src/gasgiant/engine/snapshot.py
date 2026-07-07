"""Export snapshots: an immutable copy of everything a tiled export reads.

The export spans many GUI frames; without a snapshot, mid-export slider drags
or sim stepping would make tiles 3 and 47 disagree (invisible until you get
hard tile seams — the design review's correctness blocker). All copies are
GPU-side blits; params and derived state are deep-copied on the CPU. This is
also the mechanism a future animation exporter steps through.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gasgiant.params.model import PlanetParams

if TYPE_CHECKING:
    import moderngl

    from gasgiant.gl import GpuContext
    from gasgiant.sim.vortices import VortexRegistry


def hero_centers(
    registry: VortexRegistry,
) -> list[tuple[float, float, float, float, float, float]]:
    """(x, y, z, r_core, spin, aspect) of each hero storm at its current drifted
    position. spin = sign(strength): the hero's actual rotation sense
    (seed-dependent via the ambient shear — NOT a function of hemisphere),
    which the detail pass needs to wind the analytic spiral lanes the same
    way the backtraced filaments wind."""
    out = []
    for v in registry.heroes():
        cl = math.cos(v.lat)
        out.append((
            cl * math.cos(v.lon), math.sin(v.lat), cl * math.sin(v.lon),
            v.r_core, 1.0 if v.strength >= 0.0 else -1.0, v.aspect,
        ))
    return out


@dataclass
class ExportSnapshot:
    params: PlanetParams
    tracers_eq: moderngl.Texture
    tracers_n: moderngl.Texture
    tracers_s: moderngl.Texture
    vel_eq: moderngl.Texture
    vel_n: moderngl.Texture
    vel_s: moderngl.Texture
    profile_dyn: moderngl.Texture
    profile_stamp: moderngl.Texture
    patch_rho_max: float
    blend_band: tuple[float, float]
    # Hero-storm centers at their drifted positions, (x, y, z, r_core, spin, aspect)
    # each: the detail pass amplifies/winds filaments inside them.
    heroes: list[tuple[float, float, float, float, float, float]] = None  # type: ignore[assignment]
    # Analytic lane lines and the meander warp they ride (derive-time).
    lanes: list[tuple[float, float]] = None  # type: ignore[assignment]
    warp: tuple[tuple[float, float, float], float, float] = ((0.0, 0.0, 0.0), 0.0, 3.0)
    # Field-driven detail placement: the activity texture (strain/vort) + its
    # reduction means, built from THIS snapshot's velocity so tiles agree. Both
    # None when field_drive is off (no cost for a disabled feature).
    activity_eq: moderngl.Texture = None  # type: ignore[assignment]
    activity_means: object = None  # render.activity.ActivityMeans | None

    @classmethod
    def capture(cls, sim) -> ExportSnapshot:  # sim: engine.Simulation
        gpu: GpuContext = sim.gpu
        s = sim.solver
        from gasgiant.sim.solver import BLEND_BAND, RHO_MAX

        # One equirect-velocity clone, reused for the frozen tiles AND (when
        # field_drive is on) the activity pass, so the placement matches the
        # tiles exactly. Gated build => no cost/alloc for a disabled feature.
        vel_eq = gpu.clone_texture(s.equirect.vel_tex)
        activity_eq = None
        activity_means = None
        from gasgiant.render.detail import field_drive_enabled

        if field_drive_enabled(sim.params.detail):
            from gasgiant.render.activity import ActivitySynth, new_activity_texture

            activity_eq = new_activity_texture(gpu, vel_eq.size)
            activity_means = ActivitySynth(gpu).build(vel_eq, activity_eq)

        return cls(
            params=sim.params.model_copy(deep=True),
            tracers_eq=gpu.clone_texture(s.equirect.tracers.cur),
            tracers_n=gpu.clone_texture(s.north.tracers.cur),
            tracers_s=gpu.clone_texture(s.south.tracers.cur),
            vel_eq=vel_eq,
            vel_n=gpu.clone_texture(s.north.vel_tex),
            vel_s=gpu.clone_texture(s.south.vel_tex),
            profile_dyn=gpu.clone_texture(sim.profile_dyn),
            profile_stamp=gpu.clone_texture(sim.profile_stamp),
            patch_rho_max=RHO_MAX,
            blend_band=BLEND_BAND,
            heroes=hero_centers(sim.vortices),
            lanes=list(sim.lanes),
            warp=(s.warp_offset, sim.params.bands.warp_amount, sim.params.bands.warp_freq),
            activity_eq=activity_eq,
            activity_means=activity_means,
        )

    def release(self) -> None:
        for tex in (self.tracers_eq, self.tracers_n, self.tracers_s,
                    self.vel_eq, self.vel_n, self.vel_s,
                    self.profile_dyn, self.profile_stamp):
            tex.release()
        if self.activity_eq is not None:
            self.activity_eq.release()
        if self.activity_means is not None:
            self.activity_means.release()

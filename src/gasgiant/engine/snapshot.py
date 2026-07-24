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
) -> list[tuple[float, float, float, float, float, float, float, float]]:
    """(x, y, z, r_core, spin, aspect, wake_dir, wake_lat_off) of each hero
    storm at its current drifted position. spin = sign(strength), which is the
    NEGATION of the hero's actual rotation sense (the psi-amplitude trap:
    omega = -sign(strength), see sim/vortices.py's module docstring) —
    seed-dependent via the ambient shear, not a function of hemisphere — which
    the detail pass needs to wind the analytic spiral lanes the same way the
    backtraced filaments wind. wake_dir (+1 east / -1 west / 0 no wake) and
    wake_lat_off (the wake lane's latitude offset from the hero center, rad)
    carry the sim's flow-derived wake frame to the render pass for the
    wake-braid synthesis. Consumers slice positionally and len-guard, so
    shorter tuples stay legal."""
    out = []
    for v in registry.heroes():
        cl = math.cos(v.lat)
        out.append((
            cl * math.cos(v.lon), math.sin(v.lat), cl * math.sin(v.lon),
            v.r_core, 1.0 if v.strength >= 0.0 else -1.0, v.aspect,
            v.wake_dir, v.wake_lat_off,
        ))
    return out


MAX_BRIGHT_CLOUDS = 12


def bright_cloud_centers(
    registry: VortexRegistry,
) -> list[tuple[float, float, float, float, float]]:
    """(x, y, z, r_core, aspect) of each ELONGATED bright cloud at its current
    drifted position — the companion/accent streaks the stamp renders through
    its collar-free soft-streak branch (asp > 1). The predicate mirrors that
    branch's condition, so exactly the features drawn as bright cirrus streaks
    get render-time fibers: non-hero, aspect > 1, positive brightness. Round
    storms, dark heroes, merger debris, and outbreak plumes (all aspect 1.0)
    are excluded. Capped at MAX_BRIGHT_CLOUDS in registry order (worst-case
    population is 3 heroes x 3 companions + 2 accents = 11)."""
    from gasgiant.sim.vortices import KIND_HERO

    out = []
    for v in registry.vortices:
        if v.kind == KIND_HERO or v.aspect <= 1.0 or v.brightness <= 0.0:
            continue
        cl = math.cos(v.lat)
        out.append((
            cl * math.cos(v.lon), math.sin(v.lat), cl * math.sin(v.lon),
            v.r_core, v.aspect,
        ))
        if len(out) >= MAX_BRIGHT_CLOUDS:
            break
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
    # Imported paint mask (POST art-direction), cloned so mid-export mask edits
    # can't make tiles disagree. None when no mask is bound.
    mask: moderngl.Texture | None
    patch_rho_max: float
    blend_band: tuple[float, float]
    # Hero-storm centers at their drifted positions, (x, y, z, r_core, spin,
    # aspect, wake_dir, wake_lat_off) each: the detail pass amplifies/winds
    # filaments inside them and braids their wakes.
    heroes: list[tuple[float, ...]] = None  # type: ignore[assignment]
    # Strongest EFFECTIVE hero emergence in the scene (VortexRegistry.
    # scene_emergence): the detail pass's HERO_EMERGENCE selector. Captured
    # rather than re-read from params.storms.hero_emergence at tile time,
    # because a cast hero can be emergent on its own override while the global
    # is 0 (M2-B) -- keying the render on the global there paints the LEGACY
    # hero anatomy over an emergent sim. Mixed-emergence scenes share this one
    # value in the detail pass; true per-hero detail is M2-C.
    hero_emergence: float = 0.0
    # Elongated bright-cloud centers (x, y, z, r_core, aspect) at their drifted
    # positions: the detail pass synthesizes cirrus fibers over them.
    clouds: list[tuple[float, float, float, float, float]] = None  # type: ignore[assignment]
    # Analytic lane lines and the meander warp they ride (derive-time).
    lanes: list[tuple[float, float]] = None  # type: ignore[assignment]
    warp: tuple[tuple[float, float, float], float, float] = ((0.0, 0.0, 0.0), 0.0, 3.0)

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
            vel_n=gpu.clone_texture(s.north.vel_tex),
            vel_s=gpu.clone_texture(s.south.vel_tex),
            profile_dyn=gpu.clone_texture(sim.profile_dyn),
            profile_stamp=gpu.clone_texture(sim.profile_stamp),
            mask=(gpu.clone_texture(sim._mask_tex) if sim._mask_tex is not None else None),
            patch_rho_max=RHO_MAX,
            blend_band=BLEND_BAND,
            heroes=hero_centers(sim.vortices),
            hero_emergence=sim.vortices.scene_emergence(sim.params.storms),
            clouds=bright_cloud_centers(sim.vortices),
            lanes=list(sim.lanes),
            warp=(s.warp_offset, sim.params.bands.warp_amount, sim.params.bands.warp_freq),
        )

    def release(self) -> None:
        for tex in (self.tracers_eq, self.tracers_n, self.tracers_s,
                    self.vel_eq, self.vel_n, self.vel_s,
                    self.profile_dyn, self.profile_stamp):
            tex.release()
        if self.mask is not None:
            self.mask.release()

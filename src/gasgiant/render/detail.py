"""Detail synthesis: flow-stretched filament noise + convective cells at the
output resolution, from the baked velocity and tracer textures."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext
from gasgiant.params.model import DetailParams
from gasgiant.params.seeds import subseed

if TYPE_CHECKING:
    import moderngl

_KERNELS = "gasgiant.render.kernels"
_GROUP = 16


def _set(prog: moderngl.ComputeShader, name: str, value) -> None:
    """Guarded uniform set: tolerates uniforms absent from this program
    variant (the non-DETAIL_FX program) or pruned by the driver."""
    with contextlib.suppress(KeyError):
        prog[name].value = value


@dataclass
class PolarRoute:
    """Patch velocity + tracer textures for routed polar backtraces."""

    vel_n: moderngl.Texture
    vel_s: moderngl.Texture
    tracers_n: moderngl.Texture
    tracers_s: moderngl.Texture
    rho_max: float


class DetailSynth:
    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        # Default program eagerly (its text is the pre-FX kernel, so neutral
        # defaults stay byte-identical by construction); the DETAIL_FX
        # variant compiles lazily on first selection (mirrors MapDeriver).
        self._progs: dict[bool, moderngl.ComputeShader] = {}
        self.prog = self._program(fx=False)

    def _program(self, fx: bool) -> moderngl.ComputeShader:
        if fx not in self._progs:
            defines = {"DETAIL_FX": "1"} if fx else None
            self._progs[fx] = self.gpu.compute(_KERNELS, "detail.comp", defines=defines)
        return self._progs[fx]

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
        heroes: list[tuple[float, float, float, float, float, float]] | None = None,
        polar: PolarRoute | None = None,
    ) -> None:
        """heroes: up to 3 (x, y, z, r_core, spin, aspect) hero-storm centers; the
        detail amplitude and winding time grow inside them, and the
        DETAIL_FX spiral lanes wind in the spin (= sign(strength)) sense.
        6-tuples carry aspect; shorter tuples default aspect 1.0.
        polar: patch velocity/tracer textures — when given, polar backtraces
        route through the patch charts instead of fading to neutral."""
        rng = subseed(seed, "detail-synth")
        fx_on = (
            params.intermittency > 0.0 or params.hero_spiral > 0.0
            or params.belt_texture > 0.0 or params.mottle > 0.0
            or params.belt_texture_fine > 0.0
            or params.hero_collar_wrap > 0.0
            or params.zone_texture > 0.0
        )
        prog = self._program(fx=fx_on)
        size = out_tex.size
        if fx_on:
            _set(prog, "u_intermittency", params.intermittency)
            rng_gate = subseed(seed, "detail-intermittency")
            _set(prog, "u_offset_gate", tuple(rng_gate.uniform(-100.0, 100.0, 3)))
            _set(prog, "u_hero_spiral", params.hero_spiral)
            _set(prog, "u_hero_collar_wrap", params.hero_collar_wrap)
            rng_spiral = subseed(seed, "detail-hero-spiral")
            _set(prog, "u_offset_spiral", tuple(rng_spiral.uniform(-100.0, 100.0, 3)))
            _set(prog, "u_belt_texture", params.belt_texture)
            _set(prog, "u_belt_texture_fine", params.belt_texture_fine)
            _set(prog, "u_zone_texture", params.zone_texture)
            _set(prog, "u_mottle", params.mottle)
            rng_mottle = subseed(seed, "detail-mottle")
            _set(prog, "u_offset_mottle", tuple(rng_mottle.uniform(-100.0, 100.0, 3)))
            spins = np.zeros(3, dtype=np.float32)
            for i, h in enumerate((heroes or [])[:3]):
                spins[i] = h[4] if len(h) > 4 else 1.0
            with contextlib.suppress(KeyError):
                prog["u_hero_spin"].write(spins.tobytes())
        if polar is not None:
            prog["u_polar_route"].value = 1
            polar.vel_n.use(location=3)
            prog["u_vel_n"].value = 3
            polar.vel_s.use(location=4)
            prog["u_vel_s"].value = 4
            polar.tracers_n.use(location=5)
            prog["u_tracers_n"].value = 5
            polar.tracers_s.use(location=6)
            prog["u_tracers_s"].value = 6
            prog["u_rho_max"].value = polar.rho_max
        else:
            prog["u_polar_route"].value = 0
            # Samplers must still have valid bindings.
            vel_tex.use(location=3)
            prog["u_vel_n"].value = 3
            vel_tex.use(location=4)
            prog["u_vel_s"].value = 4
            tracers_tex.use(location=5)
            prog["u_tracers_n"].value = 5
            tracers_tex.use(location=6)
            prog["u_tracers_s"].value = 6
            prog["u_rho_max"].value = 1.0
        prog["u_origin"].value = origin
        prog["u_full_size"].value = full_size if full_size is not None else size
        packed = np.zeros((3, 4), dtype=np.float32)
        aspects = np.ones(3, dtype=np.float32)   # default 1.0 -> exact short-circuit
        n_heroes = 0
        for h in (heroes or [])[:3]:
            packed[n_heroes] = h[:4]
            aspects[n_heroes] = h[5] if len(h) > 5 else 1.0
            n_heroes += 1
        prog["u_hero_count"].value = n_heroes
        prog["u_heroes"].write(packed.tobytes())
        prog["u_hero_aspect"].write(aspects.tobytes())
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
        prog["u_striation_amount"].value = params.striation_amount
        prog["u_striation_freq"].value = params.striation_frequency
        prog["u_polar_stipple"].value = params.polar_stipple
        _set(prog, "u_hero_calm", params.hero_calm)
        prog["u_offset"].value = tuple(rng.uniform(-100.0, 100.0, 3))
        out_tex.bind_to_image(0, read=False, write=True)
        gx = (size[0] + _GROUP - 1) // _GROUP
        gy = (size[1] + _GROUP - 1) // _GROUP
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

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


def _fx_param_names() -> tuple[str, ...]:
    """DETAIL_FX lever names derived from pfield metadata (``fx=True``), NOT a
    hand-maintained list: a new fx lever gets its variant-selection predicate
    and its build-time uniform tripwire by tagging the pfield (A2-6)."""
    return tuple(
        name
        for name, info in DetailParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict) and info.json_schema_extra.get("fx")
    )


_FX_PARAMS: tuple[str, ...] = _fx_param_names()


def _spread_param_names() -> tuple[str, ...]:
    """SPREAD selector lever(s) from pfield ``spread=True`` metadata."""
    return tuple(
        name
        for name, info in DetailParams.model_fields.items()
        if isinstance(info.json_schema_extra, dict)
        and info.json_schema_extra.get("spread")
    )


_SPREAD_PARAMS: tuple[str, ...] = _spread_param_names()


def spread_enabled(params: DetailParams) -> bool:
    """True when ``spread>0`` -> select the SPREAD program variant (uniform detail
    coverage). Exact zero keeps the non-variant program (byte-identical)."""
    return any(getattr(params, name) > 0.0 for name in _SPREAD_PARAMS)


def detail_fx_enabled(params: DetailParams) -> bool:
    """True when any fx-flagged lever is active -> select the DETAIL_FX program
    variant. Exact-zero on every fx lever keeps the pre-FX program (its text is
    the pre-FX kernel, so neutral defaults stay byte-identical by construction)."""
    return any(getattr(params, name) > 0.0 for name in _FX_PARAMS)


def _assert_fx_uniforms(prog) -> None:
    """A2-6 tripwire, mirroring the B1 baroclinic-uniform tripwire in
    sim/solver.py: every fx-flagged lever's ``u_<name>`` uniform must exist in
    the compiled DETAIL_FX variant. The per-dispatch ``_set`` is
    KeyError-suppressing (required: GLSL prunes unused uniforms and the non-fx
    variant legitimately lacks these), so a dropped/renamed uniform declaration
    -- or an effect block the compiler optimized away -- would make the lever
    silently inert while reporting success. Assert once at program build so any
    such regression fails loud at startup instead."""
    missing = [f"u_{name}" for name in _FX_PARAMS if _absent(prog, f"u_{name}")]
    if missing:
        raise RuntimeError(
            f"detail.comp DETAIL_FX variant is missing lever uniform(s) {missing}: "
            f"the KeyError-suppressing uniform set would silently no-op these "
            f"levers. Either the uniform declaration/effect block was dropped or "
            f"renamed, or the compiler optimized the effect out of the kernel."
        )


def _absent(prog, uniform_name: str) -> bool:
    try:
        prog[uniform_name]
    except KeyError:
        return True
    return False


def _assert_spread_uniforms(prog) -> None:
    """Tripwire mirroring _assert_fx_uniforms: the SPREAD variant must expose
    ``u_spread`` (the uniform-coverage level), else the KeyError-suppressing
    ``_set`` would silently no-op the whole effect while reporting success."""
    if _absent(prog, "u_spread"):
        raise RuntimeError(
            "detail.comp SPREAD variant is missing u_spread: the uniform-coverage "
            "blend would silently no-op. The declaration/effect block was dropped "
            "or the compiler optimized it out."
        )


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
        self._progs: dict[tuple[bool, bool], moderngl.ComputeShader] = {}
        self.prog = self._program(fx=False)

    def _program(self, fx: bool, spread: bool = False) -> moderngl.ComputeShader:
        key = (fx, spread)
        if key not in self._progs:
            defines: dict[str, str] = {}
            if fx:
                defines["DETAIL_FX"] = "1"
            if spread:
                defines["SPREAD"] = "1"
            prog = self.gpu.compute(_KERNELS, "detail.comp", defines=defines or None)
            if fx:
                _assert_fx_uniforms(prog)  # loud at build, never a silent no-op
            if spread:
                _assert_spread_uniforms(prog)
            self._progs[key] = prog
        return self._progs[key]

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
        fx_on = detail_fx_enabled(params)  # derived from pfield fx metadata
        spread_on = spread_enabled(params)  # SPREAD variant selector
        prog = self._program(fx=fx_on, spread=spread_on)
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
            _set(prog, "u_polar_filaments", params.polar_filaments)
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
        if spread_on:
            _set(prog, "u_spread", params.spread)
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

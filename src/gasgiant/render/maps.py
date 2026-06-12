"""Map derivation: tracer state -> color/height textures at any resolution.
One kernel serves the live preview and export."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.gl import GpuContext
from gasgiant.palette import bake_lut, bake_rows
from gasgiant.params.model import AppearanceParams, EmissionParams, GradientStop
from gasgiant.params.seeds import subseed

if TYPE_CHECKING:
    import moderngl

_KERNELS = "gasgiant.render.kernels"
_GROUP = 16


def _stops(stops: list[GradientStop]) -> list[tuple[float, tuple[float, float, float]]]:
    return [(s.pos, s.color) for s in stops]


def _set(prog: moderngl.ComputeShader, name: str, value) -> None:
    """Guarded uniform set: tolerates uniforms absent from this program
    variant (the non-EMISSION program) or pruned by the driver."""
    with contextlib.suppress(KeyError):
        prog[name].value = value


def chroma_uniforms(seed: int, appearance: AppearanceParams) -> dict[str, object]:
    """Pure seed-derived chroma-drift placement (fresh subseed stream) plus
    the scale/variance values; identical for the preview path, render_maps,
    and every export tile by construction."""
    rng = subseed(seed, "chroma-variance")
    offset = tuple(float(x) for x in rng.uniform(-100.0, 100.0, 3))
    return {
        "u_chroma_scale": appearance.chroma_scale,
        "u_chroma_variance": appearance.chroma_variance,
        "u_chroma_offset": offset,
    }


def emission_uniforms(seed: int, emission: EmissionParams) -> dict[str, object]:
    """Pure seed-derived emission placement: noise offsets and magnetic-pole
    unit vectors. Fresh subseed streams; identical for the preview path,
    render_maps, and every export tile by construction."""
    rng_li = subseed(seed, "emission-lightning")
    li_offset = tuple(float(x) for x in rng_li.uniform(-100.0, 100.0, 3))
    rng_au = subseed(seed, "emission-aurora")
    au_offset = tuple(float(x) for x in rng_au.uniform(-100.0, 100.0, 3))
    tilt = float(np.deg2rad(emission.aurora_pole_offset))
    lon_n = float(rng_au.uniform(-np.pi, np.pi))
    lon_s = float(rng_au.uniform(-np.pi, np.pi))
    st, ct = np.sin(tilt), np.cos(tilt)
    return {
        "u_th_color": emission.thermal_color,
        "u_th_strength": emission.thermal_strength,
        "u_th_threshold": emission.thermal_threshold,
        "u_th_hdr": emission.thermal_hdr,
        "u_li_color": emission.lightning_color,
        "u_li_strength": emission.lightning_strength,
        "u_li_density": emission.lightning_density,
        "u_li_offset": li_offset,
        "u_au_strength": emission.aurora_strength,
        "u_au_radius": float(np.deg2rad(emission.aurora_radius)),
        "u_au_width": float(np.deg2rad(emission.aurora_width)),
        "u_au_pole_n": (float(st * np.cos(lon_n)), float(ct), float(st * np.sin(lon_n))),
        "u_au_pole_s": (float(st * np.cos(lon_s)), float(-ct), float(st * np.sin(lon_s))),
        "u_au_offset": au_offset,
    }


class MapDeriver:
    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        # Program variants keyed by (EMISSION, CHROMA_FX): each disabled
        # feature preprocesses OUT of the kernel text, so neutral defaults
        # stay byte-identical by construction rather than by hope
        # (recompiling a changed kernel can shift FP scheduling on BOTH
        # sides of an off/on comparison). The two default variants compile
        # eagerly (their absence would only surface at first use);
        # FX variants compile lazily on first selection.
        self._progs: dict[tuple[bool, bool], moderngl.ComputeShader] = {}
        self.prog = self._program(emission=False, chroma_fx=False)
        self.prog_emission = self._program(emission=True, chroma_fx=False)
        self._palette_tex: moderngl.Texture | None = None
        self._storm_tex: moderngl.Texture | None = None

    def _program(self, emission: bool, chroma_fx: bool) -> moderngl.ComputeShader:
        key = (emission, chroma_fx)
        if key not in self._progs:
            defines: dict[str, str] = {}
            if emission:
                defines["EMISSION"] = "1"
            if chroma_fx:
                defines["CHROMA_FX"] = "1"
            self._progs[key] = self.gpu.compute(_KERNELS, "derive.comp", defines=defines)
        return self._progs[key]

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
        emission_out: moderngl.Texture | None = None,
        emission: EmissionParams | None = None,
        seed: int = 0,
        profile_dyn: moderngl.Texture | None = None,
        profile_stamp: moderngl.Texture | None = None,
    ) -> None:
        """lanes: (latitude, strength) thin dark lane lines; warp: the band
        meander (offset, amount, freq) the lanes ride on. Passing emission_out
        + an enabled EmissionParams selects the EMISSION program variant
        (which also needs the profile LUT textures for its gates)."""
        if self._palette_tex is None:
            self.update_palettes(appearance)
        emission_on = (
            emission_out is not None
            and emission is not None
            and emission.enabled
            and profile_dyn is not None
            and profile_stamp is not None
        )
        # Chroma FX affects the displayed color, so (unlike emission, which
        # the preview never shows) the preview uses the FX variant whenever
        # the params are active — selection depends on appearance only.
        chroma_on = (
            appearance.chroma_scale != 1.0 or appearance.chroma_variance > 0.0
        )
        prog = self._program(emission=emission_on, chroma_fx=chroma_on)
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
        prog["u_polar_tint_color"].value = appearance.polar_tint_color
        prog["u_polar_tint_strength"].value = appearance.polar_tint_strength
        prog["u_polar_tint_start"].value = float(np.deg2rad(appearance.polar_tint_start_lat))
        prog["u_contrast"].value = appearance.contrast
        prog["u_saturation"].value = appearance.saturation
        prog["u_gamma"].value = appearance.gamma
        if chroma_on:
            for name, value in chroma_uniforms(seed, appearance).items():
                _set(prog, name, value)
        color_out.bind_to_image(0, read=False, write=True)
        height_out.bind_to_image(1, read=False, write=True)
        if emission_on:
            profile_dyn.use(location=6)
            _set(prog, "u_profile_dyn", 6)
            profile_stamp.use(location=7)
            _set(prog, "u_profile_stamp", 7)
            for name, value in emission_uniforms(seed, emission).items():
                _set(prog, name, value)
            emission_out.bind_to_image(2, read=False, write=True)
        gx = (size[0] + _GROUP - 1) // _GROUP
        gy = (size[1] + _GROUP - 1) // _GROUP
        prog.run(gx, gy, 1)
        self.gpu.ctx.memory_barrier()

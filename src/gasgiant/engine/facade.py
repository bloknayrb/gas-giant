"""The Simulation facade: what the GUI, CLI, and tests consume.

Phase 3b internals: three nested domains (equirect + two polar patches)
advected in lockstep with per-step boundary exchange; the derive pass
composites them with a narrow feather. Invalidation-tier dispatch:

- RESTART  rebuild everything, re-init tracers, development run restarts
- VELOCITY rebuild jet profiles / psi uniforms, run continues (plus a few
           extra adaptation steps if it had already finished)
- POST     re-derive maps only
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.gl import GpuContext
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.render.detail import DetailSynth
from gasgiant.render.maps import MapDeriver
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.events import EventSchedule
from gasgiant.sim.profiles import build_profiles, select_wave_latitudes
from gasgiant.sim.solver import BLEND_BAND, RHO_MAX, Solver
from gasgiant.sim.vortices import generate_vortices

if TYPE_CHECKING:
    import moderngl

log = logging.getLogger(__name__)

# Extra steps to adapt after a VELOCITY-tier change once the run was finished.
_ADAPT_STEPS = 120


class Simulation:
    def __init__(self, params: PlanetParams, gpu: GpuContext | None = None) -> None:
        self.params = params
        self.gpu = gpu if gpu is not None else GpuContext.headless()
        self.deriver = MapDeriver(self.gpu)
        self.detail_synth = DetailSynth(self.gpu)
        self._preview_color: moderngl.Texture | None = None
        self._preview_height: moderngl.Texture | None = None
        self._detail_tex: moderngl.Texture | None = None
        self._post_dirty = True
        self._tracers_changed = True
        self._extra_steps = 0
        self._build()

    # -- construction / restart -------------------------------------------------

    def _build(self) -> None:
        p = self.params
        self.bands = generate_bands(p.seed, p.bands)
        self.profiles = build_profiles(p.seed, self.bands, p.bands, p.jets)
        self.vortices = generate_vortices(p.seed, self.bands, self.profiles, p.storms, p.poles)

        self.profile_dyn = self.gpu.lut_texture(self.profiles.dyn_lut())
        self.profile_stamp = self.gpu.lut_texture(self.profiles.stamp_lut())

        self.solver = Solver(
            self.gpu, p, self.profiles, self.vortices, self.profile_dyn, self.profile_stamp,
            wave_lats=select_wave_latitudes(self.bands, self.profiles),
            events=EventSchedule.generate(p, self.bands),
        )
        self.solver.init_tracers()
        self._tracers_changed = True
        self._post_dirty = True
        self._extra_steps = 0

    def _release_sim(self) -> None:
        self.solver.release()
        self.profile_dyn.release()
        self.profile_stamp.release()

    @property
    def tracers(self):
        """Equirect tracer state (tests and diagnostics)."""
        return self.solver.equirect.tracers

    # -- parameters ---------------------------------------------------------------

    def update_params(self, new_params: PlanetParams) -> set[Tier]:
        tiers = diff_tiers(self.params, new_params)
        self.params = new_params
        if Tier.RESTART in tiers:
            self._release_sim()
            self._build()
        elif Tier.VELOCITY in tiers:
            self.profiles = build_profiles(
                new_params.seed, self.bands, new_params.bands, new_params.jets
            )
            self.profile_dyn.write(self.profiles.dyn_lut().astype(np.float32).tobytes())
            self.profile_stamp.write(self.profiles.stamp_lut().astype(np.float32).tobytes())
            self.solver.params = new_params
            self.solver.set_profiles(self.profiles)
            self.solver.apply_velocity_params()
            self._extra_steps = _ADAPT_STEPS if self.is_developed else 0
            self._post_dirty = True
        elif tiers:  # POST
            self.solver.params = new_params
            self.deriver.update_palettes(new_params.appearance)
            self._post_dirty = True
        return tiers

    # -- stepping --------------------------------------------------------------------

    @property
    def steps_done(self) -> int:
        return self.solver.step_index

    @property
    def steps_target(self) -> int:
        return self.params.sim.dev_steps + self._extra_steps

    @property
    def is_developed(self) -> bool:
        return self.solver.step_index >= self.steps_target

    def tick(self, max_steps: int = 2) -> bool:
        """Advance up to max_steps of the development run. Returns True if the
        sim stepped (callers re-derive the preview)."""
        remaining = self.steps_target - self.solver.step_index
        if remaining <= 0:
            return False
        self.solver.step(min(max_steps, remaining))
        self._tracers_changed = True
        return True

    def run_to_completion(self, chunk: int = 64) -> None:
        while self.tick(chunk):
            pass

    def create_snapshot(self):
        """Immutable copy of the renderable state for tiled export."""
        from gasgiant.engine.snapshot import ExportSnapshot

        return ExportSnapshot.capture(self)

    # -- derive -----------------------------------------------------------------------

    def _get_detail_tex(self, size: tuple[int, int]) -> moderngl.Texture:
        if self._detail_tex is None or self._detail_tex.size != size:
            if self._detail_tex is not None:
                self._detail_tex.release()
            self._detail_tex = self.gpu.texture2d(size, 1, "f4", linear=True)
        return self._detail_tex

    def _derive(self, color_tex: moderngl.Texture, height_tex: moderngl.Texture) -> None:
        s = self.solver
        p = self.params
        detail_tex = None
        if p.detail.intensity > 0.0:
            detail_tex = self._get_detail_tex(color_tex.size)
            self.detail_synth.synthesize(
                p.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
                self.profile_dyn, detail_tex, p.detail,
            )
        self.deriver.derive(
            s.equirect.tracers.cur,
            s.north.tracers.cur,
            s.south.tracers.cur,
            RHO_MAX,
            BLEND_BAND,
            color_tex,
            height_tex,
            p.appearance,
            detail_tex=detail_tex,
            detail_intensity=p.detail.intensity,
        )

    # -- preview -----------------------------------------------------------------------

    def ensure_preview(self, width: int) -> tuple[moderngl.Texture, bool]:
        height = width // 2
        recreated = False
        if self._preview_color is None or self._preview_color.size != (width, height):
            if self._preview_color is not None:
                self._preview_color.release()
                self._preview_height.release()
            self._preview_color = self.gpu.texture2d((width, height), 4, "f4")
            self._preview_height = self.gpu.texture2d((width, height), 1, "f4")
            recreated = True
        if recreated or self._post_dirty or self._tracers_changed:
            self._derive(self._preview_color, self._preview_height)
            self._post_dirty = False
            self._tracers_changed = False
            return self._preview_color, True
        return self._preview_color, False

    @property
    def preview_height_texture(self) -> moderngl.Texture | None:
        return self._preview_height

    # -- export -------------------------------------------------------------------------

    def render_maps(self, width: int | None = None) -> dict[str, np.ndarray]:
        """Run the development to completion if needed, then derive maps at the
        given width and read them back. (Phase 4 replaces this with the tiled,
        detail-injected export.)"""
        self.run_to_completion()
        w = width or self.params.export.width
        color_tex = self.gpu.texture2d((w, w // 2), 4, "f4")
        height_tex = self.gpu.texture2d((w, w // 2), 1, "f4")
        try:
            self._derive(color_tex, height_tex)
            color = self.gpu.read_texture(color_tex)
            height = self.gpu.read_texture(height_tex)[..., 0]
        finally:
            color_tex.release()
            height_tex.release()
        return {"color": color, "height": height}

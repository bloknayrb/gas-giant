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
from gasgiant.sim.profiles import build_profiles, select_lanes, select_wave_latitudes
from gasgiant.sim.solver import BLEND_BAND, RHO_MAX, Solver, compute_dt
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
        # Emission preview owns its OWN scratch color/height + emission textures
        # and its OWN dirty flag, so displaying emission never perturbs the
        # Color channel's byte-identity and a POST edit can't leave stale glow.
        self._preview_emission: moderngl.Texture | None = None
        self._preview_em_color: moderngl.Texture | None = None
        self._preview_em_height: moderngl.Texture | None = None
        self._emission_preview_dirty = True
        self._detail_tex: moderngl.Texture | None = None
        self._post_dirty = True
        self._tracers_changed = True
        self._extra_steps = 0
        self._baro_driver = None
        self._baro_key: tuple | None = None
        self._baro_next_update = 0
        self._baro_update_every = 0
        self._baro_gain = 0.0
        self._baro_steps_per_update = 0
        self._baro_degraded_reason: str | None = None
        self._build()

    # -- construction / restart -------------------------------------------------

    def _build(self) -> None:
        p = self.params
        self.bands = generate_bands(p.seed, p.bands)
        self.profiles = build_profiles(p.seed, self.bands, p.bands, p.jets)
        dt = compute_dt(p.sim.resolution, p.sim.dt_scale, self.profiles.max_speed)
        self.vortices = generate_vortices(
            p.seed, self.bands, self.profiles, p.storms, p.poles,
            dt=dt,
            dev_steps=p.sim.dev_steps,
        )
        self.lanes = select_lanes(p.seed, self.bands, p.bands.lane_density)

        self.profile_dyn = self.gpu.lut_texture(self.profiles.dyn_lut())
        self.profile_stamp = self.gpu.lut_texture(self.profiles.stamp_lut())
        self.profile_omega = self.gpu.lut_texture(self.profiles.omega_lut())

        self.solver = Solver(
            self.gpu, p, self.profiles, self.vortices, self.profile_dyn, self.profile_stamp,
            wave_lats=select_wave_latitudes(self.bands, self.profiles),
            events=EventSchedule.generate(p, self.bands, self.profiles, dt),
            profile_omega_tex=self.profile_omega,
        )
        self.solver.init_tracers()
        self._init_baroclinic()
        self._tracers_changed = True
        self._post_dirty = True
        self._emission_preview_dirty = True
        self._extra_steps = 0

    def rebuild(self) -> None:
        """Deterministic full re-init from the current ``self.params``, back to
        step 0. Same release+build pairing ``update_params`` already uses for a
        RESTART-tier diff, exposed standalone so a "Restart dev" UI action can
        re-init without needing a params diff to trigger it."""
        self._release_sim()
        self._build()

    def _release_sim(self) -> None:
        if self.solver.external_omega_tex is not None:
            self.solver.external_omega_tex.release()
            self.solver.external_omega_tex = None
        self.solver.release()
        self.profile_dyn.release()
        self.profile_stamp.release()
        self.profile_omega.release()

    def _init_baroclinic(self) -> None:
        """Build/reuse the baroclinic source driver when enabled. Caches on
        (grid, warmup, seed) so unrelated RESTART edits don't re-warm. On the
        DOCUMENTED degrade signals only -- warmup outcrop (BaroclinicWarmupError)
        or a missing optional numerics dep (ImportError) -- degrade to uncoupled
        (driver=None, status 'degraded' + reason), never crash construction; a
        genuine unexpected error propagates loudly."""
        bp = self.params.solver.baroclinic
        self._baro_next_update = 0
        self._baro_update_every = bp.update_every
        self._baro_gain = bp.gain
        self._baro_steps_per_update = bp.baro_steps_per_update
        self._baro_degraded_reason = None
        if not bp.enabled:
            self._baro_driver = None
            self._baro_key = None
            return
        w, h = self.solver.equirect.size
        key = (w, h, bp.warmup_steps, self.params.seed)
        if self._baro_driver is not None and self._baro_key == key:
            self._baro_driver.reset()  # deterministic: each dev run starts post-warmup
            return  # reuse cached driver (no re-warmup)
        try:
            from gasgiant.sim.baroclinic_driver import (
                BaroclinicSourceDriver,
                BaroclinicWarmupError,
            )
            self._baro_driver = BaroclinicSourceDriver(
                grid_w=w, grid_h=h, warmup_steps=bp.warmup_steps,
                seed=self.params.seed)
            self._baro_key = key
        # ImportError FIRST: if the driver import itself failed, the
        # BaroclinicWarmupError name below was never bound.
        except ImportError as exc:
            log.warning("baroclinic coupling disabled: dependency missing (%s)", exc)
            self._degrade_baroclinic(f"dependency missing: {exc}")
        except BaroclinicWarmupError as exc:
            log.warning("baroclinic coupling disabled: warmup outcropped (%s)", exc)
            self._degrade_baroclinic(str(exc))

    def _degrade_baroclinic(self, reason: str) -> None:
        self._baro_driver = None
        self._baro_key = None
        self._baro_degraded_reason = reason

    @property
    def baroclinic_status(self) -> str:
        """``'off' | 'active' | 'degraded'``. 'degraded' means the user enabled
        baroclinic coupling but it dropped to plain uncoupled v1.6 (warmup
        outcrop, mid-run outcrop/incoherence, or a missing dependency) -- the
        GUI toasts on the transition so the degrade is never silent. See
        ``baroclinic_degraded_reason`` for the cause."""
        if not self.params.solver.baroclinic.enabled:
            return "off"
        return "active" if self._baro_driver is not None else "degraded"

    @property
    def baroclinic_degraded_reason(self) -> str | None:
        """Human-readable cause when ``baroclinic_status == 'degraded'``, else None."""
        return self._baro_degraded_reason

    def _update_baroclinic_source(self) -> None:
        """Advance the baroclinic solver and re-upload the coherent source. On
        an EXPECTED degrade (lower-layer outcrop or an incoherent/checkerboard
        source) drop to uncoupled (status 'degraded' + reason) and continue; a
        genuine unexpected error propagates loudly rather than being silently
        swallowed as a degrade."""
        from gasgiant.sim import baroclinic_source as bsrc
        from gasgiant.sim import shallow_water_ref as ref
        try:
            self._baro_driver.advance(self._baro_steps_per_update)
            src = self._baro_driver.current_source()
        except (ref.PositivityViolation, bsrc.IncoherentSourceError) as exc:
            log.warning("baroclinic source disabled mid-run: %s", exc)
            self.set_external_vorticity_source(None)
            self._degrade_baroclinic(str(exc))
            return
        self.set_external_vorticity_source(src, gain=self._baro_gain)

    @property
    def tracers(self):
        """Equirect tracer state (tests and diagnostics)."""
        return self.solver.equirect.tracers

    # -- M3 baroclinic source (opt-in external vorticity source) --------------

    def set_external_vorticity_source(
        self, field: np.ndarray | None, gain: float = 0.0
    ) -> None:
        """Bind an optional external vorticity source onto the equirect solver.

        `field` is an (H, W) or (H, W, 1) float32 array on the equirect grid
        (W, W//2); overlaid onto the Poisson RHS in omega_recover.comp each step
        (ω_rel = q − f + gain·f0·field) -- NOT into the persistent q state, so it
        is bounded and decoupled from vort_relax_tau. Pass field=None to disable.
        STRICT no-op on the default path (never called). Re-uploading a same-size
        source writes into the existing texture (no per-call alloc)."""
        if field is None:
            if self.solver.external_omega_tex is not None:
                self.solver.external_omega_tex.release()
            self.solver.external_omega_tex = None
            self.solver.external_gain = 0.0
            return
        w, h = self.solver.equirect.size
        arr = np.ascontiguousarray(field.reshape(h, w, 1).astype(np.float32))
        tex = self.solver.external_omega_tex
        if tex is None or tex.size != (w, h):
            if tex is not None:
                tex.release()
            tex = self.gpu.texture2d((w, h), 1, "f4", data=arr, linear=True)
            tex.repeat_x = True
            self.solver.external_omega_tex = tex
        else:
            tex.write(arr)  # arr is contiguous float32; avoid a per-call copy
        self.solver.external_gain = float(gain)

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
            self.lanes = select_lanes(new_params.seed, self.bands, new_params.bands.lane_density)
            self.profile_dyn.write(self.profiles.dyn_lut().astype(np.float32).tobytes())
            self.profile_stamp.write(self.profiles.stamp_lut().astype(np.float32).tobytes())
            self.profile_omega.write(self.profiles.omega_lut().astype(np.float32).tobytes())
            self.solver.params = new_params
            self.solver.set_profiles(self.profiles)
            self.solver.apply_velocity_params()
            self._extra_steps = _ADAPT_STEPS if self.is_developed else 0
            self._post_dirty = True
            self._emission_preview_dirty = True
        elif tiers:  # POST
            self.solver.params = new_params
            self._post_dirty = True
            self._emission_preview_dirty = True
        # The palette LUT depends only on appearance, but a load (or seed change)
        # edits many tiers at once. The branches above are mutually exclusive, so a
        # RESTART/VELOCITY edit that ALSO changed appearance used to skip the palette
        # refresh and leave a stale LUT -- e.g. loading the rust gas_giant_warm over a
        # paler preset kept the old colors, and toggling the solver never recolored.
        # Refresh whenever anything changed so the palette always tracks appearance.
        if tiers:
            self.deriver.update_palettes(new_params.appearance)
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
        sim stepped (callers re-derive the preview). When baroclinic coupling is
        active, the source is refreshed at fixed step_index boundaries and a step
        chunk never straddles a boundary -- so preview (small chunks) and export
        (large chunks) develop bit-identically."""
        remaining = self.steps_target - self.solver.step_index
        if remaining <= 0:
            return False
        if self._baro_driver is not None:
            if self.solver.step_index >= self._baro_next_update:
                self._update_baroclinic_source()
                self._baro_next_update += self._baro_update_every
            n = min(max_steps, remaining,
                    self._baro_next_update - self.solver.step_index)
        else:
            n = min(max_steps, remaining)
        self.solver.step(n)
        self._tracers_changed = True
        self._emission_preview_dirty = True
        return True

    def run_to_completion(self, chunk: int = 64) -> None:
        while self.tick(chunk):
            pass

    def extend_run(self, steps: int, chunk: int = 64) -> None:
        """Raise the development target by ``steps`` and advance to it (running
        any outstanding development first). Wraps the same ``_extra_steps``
        mechanism a VELOCITY-tier adaptation uses; ``tick``'s chunk-invariance
        contract makes the result independent of ``chunk``. Sequence export
        uses this to advance between frames.

        NOTE: this continues the CURRENT world — it is not equivalent to a
        fresh run with a larger ``sim.dev_steps`` (vortex census and event
        schedule are seeded from the configured dev-step count)."""
        if steps < 0:
            raise ValueError(f"extend_run steps must be >= 0, got {steps}")
        if steps == 0:
            return
        self._extra_steps += steps
        self.run_to_completion(chunk)

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

    def _derive(
        self,
        color_tex: moderngl.Texture,
        height_tex: moderngl.Texture,
        emission_tex: moderngl.Texture | None = None,
    ) -> None:
        """emission_tex=None (the Color-preview path) selects a non-EMISSION
        program variant — the viewport's Emission channel derives into its
        OWN scratch textures via ensure_preview_emission, so the Color path
        never compiles the EMISSION variant, and disabled features preprocess
        OUT of the kernel, so neutral defaults stay byte-identical by
        construction. Chroma FX is different: it affects the displayed color,
        so the deriver picks the CHROMA_FX variant from the appearance params
        alone — preview included."""
        s = self.solver
        p = self.params
        detail_tex = None
        if p.detail.intensity > 0.0:
            from gasgiant.engine.snapshot import hero_centers
            from gasgiant.render.detail import PolarRoute

            detail_tex = self._get_detail_tex(color_tex.size)
            self.detail_synth.synthesize(
                p.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
                self.profile_dyn, detail_tex, p.detail,
                heroes=hero_centers(self.vortices),
                polar=PolarRoute(
                    s.north.vel_tex, s.south.vel_tex,
                    s.north.tracers.cur, s.south.tracers.cur, RHO_MAX,
                ),
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
            lanes=self.lanes,
            warp=(s.warp_offset, p.bands.warp_amount, p.bands.warp_freq),
            emission_out=emission_tex,
            emission=p.emission if emission_tex is not None else None,
            seed=p.seed,
            profile_dyn=self.profile_dyn,
            profile_stamp=self.profile_stamp,
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

    def ensure_preview_emission(self, width: int) -> tuple[moderngl.Texture, bool]:
        """GUI preview-path emission derive, mirroring ``ensure_preview`` but
        with its OWN scratch color/height textures and its OWN dirty flag.

        The scratch color/height are throwaway (the Color channel the user
        sees still comes from ``ensure_preview``'s non-emission variant, so its
        byte-identity is preserved); only the emission texture is returned. The
        rgba32f is lazily allocated on first call (~33.5 MB @2048, then persists
        until a resize or app close -- a recorded LIMIT), so nothing is spent if
        the emission channel is never viewed.

        Always derives when dirty, regardless of ``params.emission.enabled``:
        with emission disabled the EMISSION program variant preprocesses its
        terms OUT to an all-zero map, and the GUI shows an "emission disabled"
        note off ``params.emission.enabled`` rather than relying on texture
        contents. Aurora lives in the alpha channel; the app-side viewport blit
        composites it (alpha x aurora_color) into its Emission-channel preview
        (B4-3). Returns ``(texture, rerendered)``."""
        height = width // 2
        recreated = False
        if self._preview_emission is None or self._preview_emission.size != (width, height):
            if self._preview_emission is not None:
                self._preview_emission.release()
                self._preview_em_color.release()
                self._preview_em_height.release()
            self._preview_emission = self.gpu.texture2d((width, height), 4, "f4")
            self._preview_em_color = self.gpu.texture2d((width, height), 4, "f4")
            self._preview_em_height = self.gpu.texture2d((width, height), 1, "f4")
            recreated = True
        if recreated or self._emission_preview_dirty:
            self._derive(self._preview_em_color, self._preview_em_height, self._preview_emission)
            self._emission_preview_dirty = False
            return self._preview_emission, True
        return self._preview_emission, False

    @property
    def preview_height_texture(self) -> moderngl.Texture | None:
        return self._preview_height

    @property
    def preview_tracers_texture(self) -> moderngl.Texture:
        return self.solver.equirect.tracers.cur

    # -- export -------------------------------------------------------------------------

    def render_maps(self, width: int | None = None) -> dict[str, np.ndarray]:
        """Run the development to completion if needed, then derive maps at the
        given width and read them back. (Phase 4 replaces this with the tiled,
        detail-injected export.)"""
        self.run_to_completion()
        w = width or self.params.export.width
        color_tex = self.gpu.texture2d((w, w // 2), 4, "f4")
        height_tex = self.gpu.texture2d((w, w // 2), 1, "f4")
        # Allocated only when enabled (a 16K rgba32f texture is 2 GB).
        emission_tex = (
            self.gpu.texture2d((w, w // 2), 4, "f4")
            if self.params.emission.enabled else None
        )
        try:
            self._derive(color_tex, height_tex, emission_tex)
            color = self.gpu.read_texture(color_tex)
            height = self.gpu.read_texture(height_tex)[..., 0]
            out = {"color": color, "height": height}
            if emission_tex is not None:
                out["emission"] = self.gpu.read_texture(emission_tex)
        finally:
            color_tex.release()
            height_tex.release()
            if emission_tex is not None:
                emission_tex.release()
        return out

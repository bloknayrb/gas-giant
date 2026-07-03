"""Equirect viewport: the engine's preview texture, blitted through the
selected view transform (Standard / AgX) into a display texture that imgui
shows. Channel selector switches between color, height, emission, and the
four raw tracer channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from gasgiant.gl import GpuContext

if TYPE_CHECKING:
    import moderngl

    from gasgiant.engine import Simulation

_SHADER_PACKAGE = "gasgiant.app.shaders"

CHANNELS = (
    "Color",
    "Height (cloud-top)",
    "Emission",
    "T0 color-idx",
    "T1 thickness",
    "T2 detail",
    "T3 storm-tint",
)

_EMISSION_CHANNEL = 2

# view_transform.frag's aurora-composite channel selector (B4-3): the Emission
# channel switches to it whenever aurora is on, so the aurora sliders have live
# preview feedback instead of a blind export/Blender loop.
_AURORA_COMPOSITE_CHANNEL = 5


def _emission_composite(params) -> tuple[int, tuple[float, float, float]]:
    """(u_channel, u_aurora) for the Emission channel. With aurora on, the
    preview composites the alpha-channel intensity as ``alpha * aurora_color``
    -- the same lift the Blender importer applies on its aurora shell (see
    blender_addon atmosphere.build_aurora_shell) -- so what the slider shows is
    what Blender tints. With aurora off, plain ``.rgb`` (u_channel 0), leaving
    the pre-B4-3 output untouched. Exported emission.exr semantics are
    unchanged either way: this is preview-only compositing."""
    em = params.emission
    if em.aurora_strength > 0.0:
        return _AURORA_COMPOSITE_CHANNEL, tuple(em.aurora_color)
    return 0, (0.0, 0.0, 0.0)


# (texture_selector_fn(sim, src, em_src), u_channel, force_standard_mode)
# u_channel: 0 rgb, 1 rrr, 2 ggg, 3 bbb, 4 aaa, 5 rgb+a*aurora (view_transform.frag)
# force_standard_mode=True bypasses AgX for raw diagnostic channels (AgX matrix tints grays)
_CHANNEL_MAP = [
    (lambda sim, src, em: src,                         0, False),  # Color    → .rgb
    (lambda sim, src, em: sim.preview_height_texture,  1, False),  # Height   → .rrr
    (lambda sim, src, em: em,                          0, False),  # Emission → .rgb
    (lambda sim, src, em: sim.preview_tracers_texture, 1, True),   # T0       → .r
    (lambda sim, src, em: sim.preview_tracers_texture, 2, True),   # T1       → .g
    (lambda sim, src, em: sim.preview_tracers_texture, 3, True),   # T2       → .b
    (lambda sim, src, em: sim.preview_tracers_texture, 4, True),   # T3       → .a
]


class EquirectViewport:
    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        self.pass_ = gpu.fullscreen_pass(_SHADER_PACKAGE, "view_transform.frag")
        self.channel = 0
        self.agx = False
        self._display: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._stale = True

    def mark_stale(self) -> None:
        self._stale = True

    def _ensure_display(self, size: tuple[int, int]) -> None:
        if self._display is not None and self._display.size == size:
            return
        if self._display is not None:
            self._fbo.release()
            self._display.release()
        self._display = self.gpu.texture2d(size, components=4, dtype="f1")
        self._fbo = self.gpu.framebuffer(self._display)
        self._stale = True

    def draw(self, sim: Simulation, preview_width: int) -> None:
        src, rerendered = sim.ensure_preview(preview_width)
        if rerendered:
            self._stale = True

        imgui.set_next_item_width(150.0)
        changed, self.channel = imgui.combo("channel", self.channel, list(CHANNELS))
        if changed:
            self._stale = True
        imgui.same_line()
        changed, self.agx = imgui.checkbox("AgX preview", self.agx)
        if changed:
            self._stale = True

        # Emission derives via its own preview path (separate scratch textures +
        # dirty flag) only when its channel is selected -- don't force the extra
        # ~33.5 MB derive every frame on the other channels. Aurora rides the
        # alpha channel; the blit composites it as alpha * aurora_color when it
        # is on (B4-3, see _emission_composite).
        em_src = None
        if self.channel == _EMISSION_CHANNEL:
            em_src, em_rerendered = sim.ensure_preview_emission(preview_width)
            if em_rerendered:
                self._stale = True
            if not sim.params.emission.enabled:
                imgui.text_disabled("emission disabled")
                return

        self._ensure_display(src.size)
        if self._stale:
            tex_fn, u_ch, force_std = _CHANNEL_MAP[self.channel]
            aurora = (0.0, 0.0, 0.0)
            if self.channel == _EMISSION_CHANNEL:
                u_ch, aurora = _emission_composite(sim.params)
            # Single-threaded loop: tick() completes before draw(), so tracers.cur is stable here
            source = tex_fn(sim, src, em_src)
            assert source is not None, "ensure_preview must be called before tex_fn dispatch"
            source.use(location=0)
            self.pass_.prog["u_image"].value = 0
            self.pass_.prog["u_mode"].value = 0 if force_std else (1 if self.agx else 0)
            self.pass_.prog["u_channel"].value = u_ch
            self.pass_.prog["u_aurora"].value = aurora
            self.pass_.render(self._fbo)
            self._stale = False
            # Rebind the default framebuffer: imgui's native backend renders
            # into whatever is bound, and we just bound our offscreen FBO.
            self.gpu.ctx.screen.use()

        avail = imgui.get_content_region_avail()
        w = max(avail.x, 64.0)
        h = w / 2.0
        if h > avail.y > 64.0:
            h = avail.y
            w = h * 2.0
        imgui.image(imgui.ImTextureRef(self._display.glo), imgui.ImVec2(w, h))

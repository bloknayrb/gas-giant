"""Equirect viewport: the engine's preview texture, blitted through the
selected view transform (Standard / AgX) into a display texture that imgui
shows. Channel selector switches between color and height."""

from __future__ import annotations

from typing import TYPE_CHECKING

from imgui_bundle import imgui

from gasgiant.gl import GpuContext

if TYPE_CHECKING:
    import moderngl

    from gasgiant.engine import Simulation

_SHADER_PACKAGE = "gasgiant.app.shaders"

CHANNELS = ("color", "height")


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

        imgui.set_next_item_width(120.0)
        changed, self.channel = imgui.combo("channel", self.channel, list(CHANNELS))
        if changed:
            self._stale = True
        imgui.same_line()
        changed, self.agx = imgui.checkbox("AgX preview", self.agx)
        if changed:
            self._stale = True

        self._ensure_display(src.size)
        if self._stale:
            source = src if self.channel == 0 else sim.preview_height_texture
            source.use(location=0)
            self.pass_.prog["u_image"].value = 0
            self.pass_.prog["u_mode"].value = 1 if self.agx else 0
            self.pass_.prog["u_grayscale"].value = 0 if self.channel == 0 else 1
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

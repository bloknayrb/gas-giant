"""3D sphere preview: ray-traced in a fragment shader with equirect lookup by
surface direction — per-pixel exact, no UV seam, no pole pinch. This is the
continuous QA view for seam and pole artifacts."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from imgui_bundle import imgui

from gasgiant.gl import GpuContext

if TYPE_CHECKING:
    import moderngl

_SHADER_PACKAGE = "gasgiant.app.shaders"
_SIZE = (640, 640)


class SpherePreview:
    def __init__(self, gpu: GpuContext) -> None:
        self.gpu = gpu
        self.pass_ = gpu.fullscreen_pass(_SHADER_PACKAGE, "sphere.frag")
        self.yaw = 0.6
        self.pitch = 0.25
        self.zoom = 1.0
        self._display: moderngl.Texture = gpu.texture2d(_SIZE, components=4, dtype="f1")
        self._fbo: moderngl.Framebuffer = gpu.framebuffer(self._display)

    def reset(self) -> None:
        """Restore the orbit/zoom controls to their __init__ defaults."""
        self.yaw = 0.6
        self.pitch = 0.25
        self.zoom = 1.0

    def draw(self, color_tex: moderngl.Texture, agx: bool) -> None:
        # Re-rendered every frame: a 640^2 single-bounce ray trace is trivial.
        prog = self.pass_.prog
        color_tex.use(location=0)
        prog["u_color"].value = 0
        prog["u_yaw"].value = self.yaw
        prog["u_pitch"].value = self.pitch
        prog["u_zoom"].value = self.zoom
        sun = (math.cos(0.4), 0.35, math.sin(0.4))
        norm = math.sqrt(sum(c * c for c in sun))
        prog["u_light_dir"].value = tuple(c / norm for c in sun)
        prog["u_mode"].value = 1 if agx else 0
        self.pass_.render(self._fbo)
        # Rebind the default framebuffer: imgui's native backend renders into
        # whatever is bound, and we just bound our offscreen FBO.
        self.gpu.ctx.screen.use()

        side = min(max(imgui.get_content_region_avail().x, 64.0), float(_SIZE[0]))
        imgui.image(imgui.ImTextureRef(self._display.glo), imgui.ImVec2(side, side))

        # Drag to orbit, wheel to zoom, double-click to reset, while hovering the image.
        if imgui.is_item_hovered():
            io = imgui.get_io()
            if imgui.is_mouse_dragging(0):
                self.yaw += io.mouse_delta.x * 0.01
                self.pitch = max(-1.5, min(1.5, self.pitch + io.mouse_delta.y * 0.01))
            if io.mouse_wheel:
                self.zoom = max(0.2, min(8.0, self.zoom * (1.0 + 0.12 * io.mouse_wheel)))
            if imgui.is_mouse_double_clicked(0):
                self.reset()

        imgui.text_disabled("drag: orbit · wheel: zoom · double-click: reset")

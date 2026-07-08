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

# ---------------------------------------------------------------------------
# Pure equirect <-> screen mapping helpers (T4: click-to-place / drag storms).
# The displayed preview is an equirect map: x spans longitude -180..+180
# left->right, y spans latitude +90..-90 top->bottom. `rect_min`/`rect_max` are
# the (x, y) pixel corners of the on-screen image item (imgui.get_item_rect_*).
# These are pure and side-effect free so main.py's tool wiring is unit-testable
# without a GL context or a live imgui frame.
# ---------------------------------------------------------------------------


def screen_to_lonlat(
    mx: float,
    my: float,
    rect_min: tuple[float, float],
    rect_max: tuple[float, float],
) -> tuple[float, float]:
    """Map a mouse pixel inside the displayed equirect image to ``(lon, lat)``
    degrees. The result is clamped to lon -180..+180 / lat -90..+90, so a click
    outside the rect maps to the nearest valid edge point."""
    x0, y0 = rect_min
    x1, y1 = rect_max
    w = x1 - x0
    h = y1 - y0
    fx = 0.0 if w == 0 else (mx - x0) / w
    fy = 0.0 if h == 0 else (my - y0) / h
    fx = min(1.0, max(0.0, fx))
    fy = min(1.0, max(0.0, fy))
    lon = -180.0 + fx * 360.0
    lat = 90.0 - fy * 180.0
    return lon, lat


def lonlat_to_screen(
    lon_deg: float,
    lat_deg: float,
    rect_min: tuple[float, float],
    rect_max: tuple[float, float],
) -> tuple[float, float]:
    """Inverse of ``screen_to_lonlat`` (for drawing markers). NOT clamped: a
    wrapped duplicate longitude (entry lon +-360) maps outside the rect on
    purpose so the caller can clip it and show the dateline wrap."""
    x0, y0 = rect_min
    x1, y1 = rect_max
    fx = (lon_deg + 180.0) / 360.0
    fy = (90.0 - lat_deg) / 180.0
    sx = x0 + fx * (x1 - x0)
    sy = y0 + fy * (y1 - y0)
    return sx, sy


def nearest_cast_index(lon_deg, lat_deg, cast, max_deg=None):
    """Index of the cast entry nearest ``(lon_deg, lat_deg)`` using a wrap-aware
    longitude metric (shortest arc across the +-180 dateline) combined with the
    latitude delta. ``cast`` is a list of objects carrying ``.lat_deg``/
    ``.lon_deg``. Returns None if ``cast`` is empty or, when ``max_deg`` is
    given, nothing lies within ``max_deg`` of the point."""
    best_i = None
    best_d = None
    for i, entry in enumerate(cast):
        dlon = abs(((lon_deg - entry.lon_deg + 180.0) % 360.0) - 180.0)
        dlat = abs(lat_deg - entry.lat_deg)
        d = (dlon * dlon + dlat * dlat) ** 0.5
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    if best_i is None:
        return None
    if max_deg is not None and best_d > max_deg:
        return None
    return best_i


# Per-kind marker (fill color rgb, glyph). Kinds are CastKind values.
_KIND_MARKER: dict[str, tuple[tuple[float, float, float], str]] = {
    "hero": ((0.95, 0.42, 0.24), "H"),
    "oval": ((0.96, 0.90, 0.70), "O"),
    "barge": ((0.72, 0.46, 0.30), "B"),
    "pearl": ((0.88, 0.94, 1.00), "P"),
}
_MARKER_RADIUS = 6.0

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
        # T5 A/B compare: snapshot A gets its OWN view-transformed display
        # texture so the split/flash compare is apples-to-apples (same AgX pass
        # as the live side). _display_a_glo pins which snapshot is currently
        # baked in, so a fresh snapshot re-renders even when nothing else moved.
        self._display_a: moderngl.Texture | None = None
        self._fbo_a: moderngl.Framebuffer | None = None
        self._display_a_glo: int | None = None
        self._compare_stale = True
        # (x, y) pixel corners of the on-screen equirect image, captured after
        # imgui.image() each frame so main.py can hit-test the storm tool. None
        # whenever no image was drawn this frame (e.g. emission disabled).
        self.image_rect_min: tuple[float, float] | None = None
        self.image_rect_max: tuple[float, float] | None = None
        # True only for the normal single-image equirect (a hit-testable map for
        # the storm tool); False during an A/B compare split/flash view.
        self.hit_testable = False

    def mark_stale(self) -> None:
        self._stale = True
        self._compare_stale = True

    def _ensure_display(self, size: tuple[int, int]) -> None:
        if self._display is not None and self._display.size == size:
            return
        if self._display is not None:
            self._fbo.release()
            self._display.release()
        self._display = self.gpu.texture2d(size, components=4, dtype="f1")
        self._fbo = self.gpu.framebuffer(self._display)
        self._stale = True

    def draw(
        self,
        sim: Simulation,
        preview_width: int,
        *,
        compare_mode: str = "off",
        snapshot_a: moderngl.Texture | None = None,
        flash_show_a: bool = False,
    ) -> None:
        # Reset each frame: an early return (emission disabled) leaves no image,
        # so the tool must not hit-test against a stale rect.
        self.image_rect_min = None
        self.image_rect_max = None
        self.hit_testable = False
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
            self._compare_stale = True  # snapshot A shares the AgX view transform

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

        # T5 A/B compare (only when a mode is chosen AND snapshot A is held).
        # A split/flash view is NOT a hit-testable equirect: image_rect_* is set
        # (so the app's imgui block still runs) but hit_testable stays False, so
        # the storm tool disengages.
        if compare_mode != "off" and snapshot_a is not None:
            self._render_compare_a(snapshot_a)
            p0 = imgui.get_cursor_screen_pos()
            if compare_mode == "split":
                self._draw_compare_split(w, h)
            else:  # "flash": whole image, toggled between A and live
                self._draw_compare_single(
                    self._display_a if flash_show_a else self._display,
                    "A (snapshot)" if flash_show_a else "live",
                    w, h,
                )
            self.image_rect_min = (p0.x, p0.y)
            self.image_rect_max = (p0.x + w, p0.y + h)
            return

        imgui.image(imgui.ImTextureRef(self._display.glo), imgui.ImVec2(w, h))
        rmin = imgui.get_item_rect_min()
        rmax = imgui.get_item_rect_max()
        self.image_rect_min = (rmin.x, rmin.y)
        self.image_rect_max = (rmax.x, rmax.y)
        self.hit_testable = True

    # -- T5 A/B compare rendering ------------------------------------------------

    def _ensure_display_a(self, size: tuple[int, int]) -> None:
        if self._display_a is not None and self._display_a.size == size:
            return
        if self._display_a is not None:
            self._fbo_a.release()
            self._display_a.release()
        self._display_a = self.gpu.texture2d(size, components=4, dtype="f1")
        self._fbo_a = self.gpu.framebuffer(self._display_a)
        self._compare_stale = True

    def _render_compare_a(self, snapshot_a: moderngl.Texture) -> None:
        """View-transform the held snapshot-A color texture into its OWN display
        texture through the SAME pass (and current AgX mode) the live side uses,
        so the compare is apples-to-apples. Snapshot A is always the Color
        channel -- it is a clone of the color preview. Re-renders only when the
        view went stale or a fresh snapshot was taken. This binds an offscreen
        FBO, so it rebinds the default framebuffer before returning (imgui's
        native backend renders into whatever is bound)."""
        self._ensure_display_a(snapshot_a.size)
        if not self._compare_stale and self._display_a_glo == snapshot_a.glo:
            return
        snapshot_a.use(location=0)
        self.pass_.prog["u_image"].value = 0
        self.pass_.prog["u_mode"].value = 1 if self.agx else 0
        self.pass_.prog["u_channel"].value = 0
        self.pass_.prog["u_aurora"].value = (0.0, 0.0, 0.0)
        self.pass_.render(self._fbo_a)
        self._compare_stale = False
        self._display_a_glo = snapshot_a.glo
        self.gpu.ctx.screen.use()  # rebind default framebuffer for imgui

    def _draw_compare_split(self, w: float, h: float) -> None:
        """Vertical split: left half = snapshot A, right half = live. Each half
        samples its OWN half of the source via uv0/uv1 (imgui-bundle 1.92's
        imgui.image takes uv0/uv1 directly), so there is no horizontal squash."""
        half = w * 0.5
        p0 = imgui.get_cursor_screen_pos()
        imgui.image(
            imgui.ImTextureRef(self._display_a.glo), imgui.ImVec2(half, h),
            imgui.ImVec2(0.0, 0.0), imgui.ImVec2(0.5, 1.0),
        )
        imgui.same_line(0.0, 0.0)
        imgui.image(
            imgui.ImTextureRef(self._display.glo), imgui.ImVec2(half, h),
            imgui.ImVec2(0.5, 0.0), imgui.ImVec2(1.0, 1.0),
        )
        draw_list = imgui.get_window_draw_list()
        mid = p0.x + half
        line = imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 0.9))
        draw_list.add_line(imgui.ImVec2(mid, p0.y), imgui.ImVec2(mid, p0.y + h), line, 1.5)
        self._overlay_label(draw_list, p0.x + 6.0, p0.y + 6.0, "A (snapshot)")
        self._overlay_label(draw_list, mid + 6.0, p0.y + 6.0, "live")

    def _draw_compare_single(
        self, display: moderngl.Texture | None, label: str, w: float, h: float
    ) -> None:
        p0 = imgui.get_cursor_screen_pos()
        imgui.image(imgui.ImTextureRef(display.glo), imgui.ImVec2(w, h))
        draw_list = imgui.get_window_draw_list()
        self._overlay_label(draw_list, p0.x + 6.0, p0.y + 6.0, f"showing: {label}")

    @staticmethod
    def _overlay_label(draw_list, x: float, y: float, text: str) -> None:
        bg = imgui.get_color_u32(imgui.ImVec4(0.0, 0.0, 0.0, 0.55))
        fg = imgui.get_color_u32(imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
        ts = imgui.calc_text_size(text)
        draw_list.add_rect_filled(
            imgui.ImVec2(x - 3.0, y - 2.0),
            imgui.ImVec2(x + ts.x + 3.0, y + ts.y + 2.0), bg,
        )
        draw_list.add_text(imgui.ImVec2(x, y), fg, text)

    def draw_markers(self, cast, *, drag_index=None, drag_lonlat=None) -> None:
        """Overlay a marker (color-coded circle + kind glyph) at each cast
        entry's rendered position using the window draw list. Must be called in
        the same imgui window as ``draw`` (right after it). Near the +-180 edges
        the wrapped duplicates (lon +-360) are drawn too and clipped to the image
        rect, so a dateline-straddling storm shows on both sides. The entry being
        dragged is drawn at ``drag_lonlat`` (its live cursor position) and
        highlighted."""
        if self.image_rect_min is None or self.image_rect_max is None:
            return
        rmin = self.image_rect_min
        rmax = self.image_rect_max
        draw_list = imgui.get_window_draw_list()
        draw_list.push_clip_rect(
            imgui.ImVec2(*rmin), imgui.ImVec2(*rmax), True
        )
        # margin so a glyph whose center is just off the edge still paints its
        # visible (wrapped) half inside the rect.
        left = rmin[0] - _MARKER_RADIUS - 2.0
        right = rmax[0] + _MARKER_RADIUS + 2.0
        for i, entry in enumerate(cast):
            if drag_index == i and drag_lonlat is not None:
                lon, lat = drag_lonlat
            else:
                lon, lat = entry.lon_deg, entry.lat_deg
            color, glyph = _KIND_MARKER.get(str(entry.kind), ((1.0, 1.0, 1.0), "?"))
            highlighted = drag_index == i
            for lon_variant in (lon, lon - 360.0, lon + 360.0):
                sx, sy = lonlat_to_screen(lon_variant, lat, rmin, rmax)
                if left <= sx <= right:
                    self._draw_marker(draw_list, sx, sy, color, glyph, highlighted)
        draw_list.pop_clip_rect()

    @staticmethod
    def _draw_marker(draw_list, sx, sy, color, glyph, highlighted) -> None:
        center = imgui.ImVec2(sx, sy)
        fill = imgui.get_color_u32(imgui.ImVec4(color[0], color[1], color[2], 0.9))
        outline = imgui.get_color_u32(
            imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
            if highlighted
            else imgui.ImVec4(0.0, 0.0, 0.0, 0.85)
        )
        r = _MARKER_RADIUS + (1.5 if highlighted else 0.0)
        draw_list.add_circle_filled(center, r, fill, 16)
        draw_list.add_circle(center, r, outline, 16, 2.0 if highlighted else 1.5)
        text_col = imgui.get_color_u32(imgui.ImVec4(0.05, 0.05, 0.05, 1.0))
        ts = imgui.calc_text_size(glyph)
        draw_list.add_text(
            imgui.ImVec2(sx - ts.x * 0.5, sy - ts.y * 0.5), text_col, glyph
        )

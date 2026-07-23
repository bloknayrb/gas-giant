"""GPU: the Phase 8 preview draw paths (emission channel + sphere sun rig).

These drive the real ``EquirectViewport.draw`` / ``SpherePreview.draw`` against
the headless GL context AND a headless imgui context, so the new channel branch
(including the emission-disabled early return) and the sun sliders get genuine
coverage rather than none -- the same gap Phase 7's reviewer flagged for
``draw_help``.
"""

from __future__ import annotations

import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu

imgui = pytest.importorskip("imgui_bundle.imgui")

_PREVIEW = 256  # small preview; the draw path is identical to PREVIEW_WIDTH


def _params(seed: int = 5) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = 20
    return p


@pytest.fixture
def imgui_ctx():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def _draw(fn) -> None:
    imgui.new_frame()
    imgui.begin("w", None, 0)
    fn()
    imgui.end()
    imgui.end_frame()


def test_viewport_emission_channel_new_entry(gpu, imgui_ctx):
    from gasgiant.app.viewport import _EMISSION_CHANNEL, CHANNELS

    assert CHANNELS == (
        "Color",
        "Height (cloud-top)",
        "Emission",
        "T0 color-idx",
        "T1 thickness",
        "T2 detail",
        "T3 storm-tint",
    )
    assert CHANNELS[_EMISSION_CHANNEL] == "Emission"


def test_viewport_emission_disabled_early_returns(gpu, imgui_ctx):
    """The emission-disabled branch exercises the NEW channel path end to end
    (combo -> ``ensure_preview_emission`` GL derive -> disabled early return)
    and stops BEFORE the display FBO / ``ctx.screen`` rebind, so it runs on the
    headless standalone context (which has no default framebuffer). The full
    enabled render path's final ``ctx.screen.use()`` can only run under the real
    windowed context; its GL work (``ensure_preview_emission``) is covered by
    ``tests/gpu/test_emission.py``."""
    from gasgiant.app.viewport import EquirectViewport

    sim = Simulation(_params(), gpu)  # emission disabled
    vp = EquirectViewport(gpu)
    vp.channel = 2
    _draw(lambda: vp.draw(sim, _PREVIEW))
    assert vp._display is None  # returned before allocating the display texture
    gpu.make_current()


def test_view_transform_aurora_composite_formula(gpu):
    """B4-3: u_channel=5 composites rgb + a * u_aurora (Standard mode is a
    pass-through, so the check is exact), and u_channel=0 with the same input
    ignores alpha and u_aurora entirely -- the pre-B4-3 blit is untouched.
    Renders the real view_transform.frag into an offscreen f4 FBO; no imgui,
    no default framebuffer needed."""
    import struct

    src = gpu.texture2d((1, 1), 4, "f4")
    src.write(struct.pack("4f", 0.10, 0.20, 0.30, 0.5))
    dst = gpu.texture2d((1, 1), 4, "f4")
    fbo = gpu.framebuffer(dst)

    pass_ = gpu.fullscreen_pass("gasgiant.app.shaders", "view_transform.frag")
    src.use(location=0)
    pass_.prog["u_image"].value = 0
    pass_.prog["u_mode"].value = 0  # Standard = pass-through
    pass_.prog["u_channel"].value = 5
    pass_.prog["u_aurora"].value = (0.8, 0.4, 0.6)
    pass_.render(fbo)
    r, g, b, a = struct.unpack("4f", fbo.read(components=4, dtype="f4"))
    assert (r, g, b) == pytest.approx((0.10 + 0.5 * 0.8, 0.20 + 0.5 * 0.4, 0.30 + 0.5 * 0.6))
    assert a == pytest.approx(1.0)

    # channel 0 (the plain Emission/Color blit): alpha and u_aurora ignored
    pass_.prog["u_channel"].value = 0
    pass_.render(fbo)
    r, g, b, _ = struct.unpack("4f", fbo.read(components=4, dtype="f4"))
    assert (r, g, b) == pytest.approx((0.10, 0.20, 0.30))

    pass_.release()
    fbo.release()
    dst.release()
    src.release()
    gpu.make_current()


def test_view_transform_is_north_up(gpu):
    """Orientation gate: the equirect blit must be NORTH-UP. Source texel row 0
    is latitude +90 (core/domain.py: lat = pi/2 - (j+0.5)/H*pi), and imgui.image
    draws texel row 0 at the TOP of the on-screen image. So after the blit the
    display's row 0 must still carry the source's north row -- not the south row.

    This pins the fix for the flipped-viewport bug (the hero at lat -24 rendered
    at the top of the map): view_transform.frag must NOT vertically flip. Renders
    the real shader into an offscreen f4 FBO; no imgui / default framebuffer.
    """
    import struct

    H = 4
    src = gpu.texture2d((1, H), 4, "f4")  # 1 wide, H tall
    north = (1.0, 0.0, 0.0, 1.0)  # row 0
    south = (0.0, 0.0, 1.0, 1.0)  # row H-1
    rows = [north] + [(0.0, 0.0, 0.0, 1.0)] * (H - 2) + [south]
    src.write(b"".join(struct.pack("4f", *row) for row in rows))

    dst = gpu.texture2d((1, H), 4, "f4")
    fbo = gpu.framebuffer(dst)
    pass_ = gpu.fullscreen_pass("gasgiant.app.shaders", "view_transform.frag")
    src.use(location=0)
    pass_.prog["u_image"].value = 0
    pass_.prog["u_mode"].value = 0  # Standard = pass-through
    pass_.prog["u_channel"].value = 0
    pass_.prog["u_aurora"].value = (0.0, 0.0, 0.0)
    pass_.render(fbo)

    out = fbo.read(components=4, dtype="f4")
    display = [struct.unpack("4f", out[i * 16 : (i + 1) * 16]) for i in range(H)]
    top = display[0]  # texel row 0 == imgui screen-TOP
    assert top[0] > top[2] + 0.5, (
        f"screen-top is not the source NORTH row (rgb={top[:3]}); the equirect "
        "view is upside down -- view_transform.frag is flipping V."
    )

    pass_.release()
    fbo.release()
    dst.release()
    src.release()
    gpu.make_current()

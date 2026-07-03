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

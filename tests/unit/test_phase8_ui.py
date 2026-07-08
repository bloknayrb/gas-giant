"""Phase 8 UI: export modal + sphere sun-direction rig.

The export-modal draw path and the sphere ``reset`` are exercised against a
real (headless) imgui context -- the same ``imgui.create_context()`` +
``new_frame``/``end_frame`` pattern ``test_shortcuts.py`` uses -- so the new
draw code has genuine coverage rather than none (Phase 7's reviewer flagged
``draw_help`` for exactly this gap). No GL context is needed: the modal issues
no moderngl calls, and ``SpherePreview.reset`` is pure state.
"""

from __future__ import annotations

import pytest

from gasgiant.params.model import PlanetParams

main = pytest.importorskip("gasgiant.app.main")
sphere_mod = pytest.importorskip("gasgiant.app.sphere_preview")
imgui = pytest.importorskip("imgui_bundle.imgui")

StudioApp = main.StudioApp
SpherePreview = sphere_mod.SpherePreview


@pytest.fixture
def imgui_ctx():
    ctx = imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(800.0, 600.0)
    io.delta_time = 1.0 / 60.0
    io.set_ini_filename(None)  # don't litter the repo root with an imgui.ini
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield imgui
    imgui.destroy_context(ctx)


def _make_export_app(emission_enabled: bool) -> StudioApp:
    params = PlanetParams()
    if emission_enabled:
        params.emission.thermal_strength = 0.5
    else:
        params.emission.thermal_strength = 0.0
        params.emission.lightning_strength = 0.0
        params.emission.aurora_strength = 0.0
    app = StudioApp.__new__(StudioApp)
    app.params = params
    app._live = params
    app._dialog = None
    app._export = None
    app._seq_frames = 1
    app._seq_steps_per_frame = 30
    app._seq_all_maps = False
    app._seq_video = False
    app._seq_fps = 24
    app._restart_dev_after_export = False
    app._export_is_sequence = False
    app._commit = lambda new_params: setattr(app, "params", new_params)  # type: ignore[method-assign]
    app._reset_working_copy = lambda: None  # type: ignore[method-assign]
    return app


def _draw_modal(app: StudioApp) -> None:
    imgui.new_frame()
    imgui.begin("w", None, 0)
    imgui.open_popup("Export map set")
    app._draw_export_modal()
    imgui.end()
    imgui.end_frame()


@pytest.mark.parametrize("emission_enabled", [True, False])
def test_export_modal_draws_without_crashing(imgui_ctx, emission_enabled):
    """The modal body (resolution combo + clarifier, compression slider,
    emission indicator, Export/Cancel) renders for both emission states."""
    app = _make_export_app(emission_enabled)
    _draw_modal(app)
    # No side effect from merely drawing the modal: the folder picker only
    # fires from an explicit click on the inner Export button.
    assert app._dialog is None


def test_export_modal_emission_indicator_reads_enabled_property(imgui_ctx):
    """Sanity that the indicator's condition tracks the params: enabling any
    emission term flips ``params.emission.enabled`` (the value the modal
    branches on)."""
    on = _make_export_app(True)
    off = _make_export_app(False)
    assert on.params.emission.enabled is True
    assert off.params.emission.enabled is False
    _draw_modal(on)
    _draw_modal(off)


class _FakeUniform:
    def __init__(self) -> None:
        self.value = None


class _FakeProg:
    def __init__(self) -> None:
        self._u: dict[str, _FakeUniform] = {}

    def __getitem__(self, key: str) -> _FakeUniform:
        return self._u.setdefault(key, _FakeUniform())


class _FakePass:
    def __init__(self) -> None:
        self.prog = _FakeProg()
        self.rendered = 0

    def render(self, _fbo) -> None:
        self.rendered += 1


class _FakeCtx:
    class _Screen:
        def use(self) -> None:
            pass

    screen = _Screen()


class _FakeGpu:
    ctx = _FakeCtx()


class _FakeTex:
    glo = 1

    def use(self, location: int = 0) -> None:
        pass


def _make_sphere() -> SpherePreview:
    sp = SpherePreview.__new__(SpherePreview)
    sp.gpu = _FakeGpu()
    sp.pass_ = _FakePass()
    sp._fbo = object()
    sp._display = _FakeTex()
    sp.reset()  # seeds yaw/pitch/zoom + sun defaults
    return sp


def test_sphere_draw_sets_light_dir_from_sun_angles(imgui_ctx):
    """The sun sliders feed spherical coords into ``u_light_dir`` as a unit
    vector; the default rig reproduces the former hardcoded ~0.34 vertical
    component. Driven with a fake GL pass so it runs headless."""
    import math

    sp = _make_sphere()
    tex = _FakeTex()

    imgui.new_frame()
    imgui.begin("w", None, 0)
    sp.draw(tex, agx=False)
    imgui.end()
    imgui.end_frame()

    lx, ly, lz = sp.pass_.prog["u_light_dir"].value
    assert math.isclose(lx * lx + ly * ly + lz * lz, 1.0, rel_tol=1e-6)
    assert math.isclose(ly, math.sin(0.35), abs_tol=1e-6)  # vertical ~0.343
    assert lx > 0.0 and lz > 0.0  # azimuth 0.4 default quadrant


def test_sphere_reset_restores_sun_direction():
    """double-click reset restores the sun rig to its defaults alongside
    yaw/pitch/zoom -- no GL needed, ``reset`` is pure state."""
    sp = SpherePreview.__new__(SpherePreview)
    sp.yaw = 3.0
    sp.pitch = 1.0
    sp.zoom = 4.0
    sp.sun_azimuth = -2.0
    sp.sun_elevation = 1.2
    sp.reset()
    assert (sp.yaw, sp.pitch, sp.zoom) == (0.6, 0.25, 1.0)
    assert sp.sun_azimuth == 0.4
    assert sp.sun_elevation == 0.35

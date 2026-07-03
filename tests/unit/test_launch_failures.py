"""B1-2: friendly startup-failure messages.

Two first-minute failure modes used to surface as raw tracebacks:
- plain `uv sync` (no GUI extra) installs the gasgiant-studio entry point,
  which died with a bare ImportError for imgui_bundle;
- a GPU below OpenGL 4.3 propagated an unhandled exception from init_gl.

The launcher tests need NO GUI extra (gasgiant.app.launcher deliberately has
no imgui import at module scope); the init_gl tests drive StudioApp with the
same headless fakes test_shortcuts.py uses.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from gasgiant.app import launcher

# -- launcher: missing-GUI-extra translation ------------------------------------


def test_launcher_module_has_no_gui_imports():
    """The whole point of the launcher: it must be importable in an
    environment without the GUI extra, so no module-scope imgui import."""
    tree = ast.parse(inspect.getsource(launcher))
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.add(node.module or "")
    assert not any("imgui" in name for name in top_level_imports)
    assert not any(name.startswith("gasgiant.app.main") for name in top_level_imports)


def test_missing_gui_message_for_imgui_bundle():
    msg = launcher.missing_gui_message(ImportError("nope", name="imgui_bundle"))
    assert msg is not None
    assert "uv sync --all-extras" in msg
    assert "imgui-bundle" in msg


def test_missing_gui_message_for_imgui_bundle_submodule():
    msg = launcher.missing_gui_message(ImportError("nope", name="imgui_bundle.imgui"))
    assert msg is not None
    assert "uv sync --all-extras" in msg


def test_missing_gui_message_none_for_other_imports():
    """An unrelated ImportError must keep its traceback -- an 'install the
    extra' hint for a broken numpy would send the user chasing the wrong fix."""
    assert launcher.missing_gui_message(ImportError("nope", name="numpy")) is None
    assert launcher.missing_gui_message(ImportError("nope")) is None


def test_launcher_main_prints_hint_and_exits_1(monkeypatch, capsys):
    def boom():
        raise ImportError("No module named 'imgui_bundle'", name="imgui_bundle")

    monkeypatch.setattr(launcher, "_import_studio_main", boom)
    assert launcher.main() == 1
    err = capsys.readouterr().err
    assert "uv sync --all-extras" in err
    assert "Traceback" not in err


def test_launcher_main_reraises_unrelated_import_error(monkeypatch):
    def boom():
        raise ImportError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr(launcher, "_import_studio_main", boom)
    with pytest.raises(ImportError, match="numpy"):
        launcher.main()


def test_launcher_main_runs_studio_main(monkeypatch):
    monkeypatch.setattr(launcher, "_import_studio_main", lambda: (lambda: 0))
    assert launcher.main() == 0


def test_entry_point_routes_through_launcher():
    """pyproject must point gasgiant-studio at the launcher, not app.main --
    otherwise the module-scope imgui import fires before any translation."""
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'gasgiant-studio = "gasgiant.app.launcher:main"' in text


# -- init_gl: GL < 4.3 / attach failure ------------------------------------------

main_mod = pytest.importorskip("gasgiant.app.main")


def test_gl_failure_message_is_actionable():
    msg = main_mod._gl_failure_message("context creation failed")
    assert "OpenGL 4.3" in msg
    assert "LIBGL_ALWAYS_SOFTWARE" in msg
    assert "context creation failed" in msg


class _FakeRunnerParams:
    app_shall_exit = False


def _make_init_app() -> object:
    app = main_mod.StudioApp.__new__(main_mod.StudioApp)
    app.gpu = None
    app.sim = None
    app.viewport = None
    app.sphere = None
    app.init_error = None
    app._last_export_dir = None
    from gasgiant.params.model import PlanetParams

    app.params = PlanetParams()
    return app


def test_init_gl_attach_failure_sets_friendly_error(monkeypatch):
    fake_runner = _FakeRunnerParams()
    monkeypatch.setattr(main_mod.hello_imgui, "get_runner_params", lambda: fake_runner)

    class BoomContext:
        @classmethod
        def attach(cls):
            raise RuntimeError("no current GL context")

    monkeypatch.setattr(main_mod, "GpuContext", BoomContext)
    app = _make_init_app()
    app.init_gl()  # must NOT raise (a raise through the native callback = raw traceback)
    assert app.init_error is not None
    assert "OpenGL 4.3" in app.init_error
    assert "no current GL context" in app.init_error
    assert fake_runner.app_shall_exit is True
    assert app.sim is None


def test_init_gl_version_below_43_sets_friendly_error(monkeypatch):
    fake_runner = _FakeRunnerParams()
    monkeypatch.setattr(main_mod.hello_imgui, "get_runner_params", lambda: fake_runner)

    class FakeCtx:
        version_code = 330
        info = {"GL_RENDERER": "old-gpu"}

    class FakeGpu:
        ctx = FakeCtx()

        @classmethod
        def attach(cls):
            return cls()

    monkeypatch.setattr(main_mod, "GpuContext", FakeGpu)
    app = _make_init_app()
    app.init_gl()
    assert app.init_error is not None
    assert "3.3" in app.init_error  # names what it actually got
    assert "4.3" in app.init_error
    assert fake_runner.app_shall_exit is True


def test_draw_callbacks_survive_failed_init(monkeypatch):
    """After a failed init the runner may still draw a frame; every dock-pane
    callback must be a no-op (Controls shows the message) instead of an
    AttributeError on sim=None."""
    imgui = pytest.importorskip("imgui_bundle.imgui")
    ctx = imgui.create_context()
    try:
        io = imgui.get_io()
        io.display_size = imgui.ImVec2(800.0, 600.0)
        io.delta_time = 1.0 / 60.0
        io.set_ini_filename(None)
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures

        app = _make_init_app()
        app.init_error = main_mod._gl_failure_message("boom")
        imgui.new_frame()
        imgui.begin("w", None, 0)
        app.draw_controls()
        app.draw_equirect()
        app.draw_sphere()
        app.draw_perf()
        imgui.end()
        imgui.end_frame()
    finally:
        imgui.destroy_context(ctx)
